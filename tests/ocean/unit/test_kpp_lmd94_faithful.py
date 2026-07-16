"""KPP (Large, McWilliams & Doney 1994) ORACLE-FAITHFULNESS tests.

The ocean KPP boundary-layer scheme (``ocean/physics/vertical_mixing/kpp.py``) is
a faithful LMD94 implementation, but the existing tests only check directional /
shape / config-liveness / conservation properties — none pin the LMD94 closed
forms to round-off.  These pin them against an INDEPENDENT reimplementation whose
constants are typed from LMD94 Appendix B, and canary the ``KPPConfig`` defaults.

What is pinned, in order of authority:
1. ``_kpp_velocity_scales`` w_m/w_s to rel 1e-12 across a (u*, B_f, d, h) grid
   spanning ALL THREE regimes (stable, weakly unstable, convective); CONSTRUCTED
   cases pin the momentum/scalar joins at |zeta| = zeta_m = 0.2 / zeta_s = 1.0
   (with branch-identity asserts) + the surface-layer cap d_eff = min(d, eps*h).
2. ``_kpp_unresolved_shear_variance`` V_t^2 (LMD94 Eq. 23) to round-off + its
   linearity + the explicit sqrt(-beta_T) prefactor (the fixed 2.236x bug).
3. ``_kpp_shape_function`` G(sigma) = sigma*(1-sigma)^2 + the DEPARTURE canary
   G(1) = G'(1) = 0 (reduced non-matching cubic; under-represented entrainment).
4. ``KPPConfig`` App. B coefficient canary vs the published LMD94 values.
5. x64 AD-finiteness of the velocity scales, V_t^2, and G.

Precision: the scales carry no precision-policy promotion; the autouse fixture
enables x64 for the round-off pins and restores the entry state.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.physics.vertical_mixing.kpp import (
    _EPS,
    _kpp_shape_function,
    _kpp_unresolved_shear_variance,
    _kpp_velocity_scales,
)


@pytest.fixture(autouse=True)
def _force_x64():
    entry_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", entry_x64)

# LMD94 Appendix B constants, typed from the paper (NOT copied from KPPConfig);
# the canary below asserts the config defaults equal these.
_O_KAPPA = 0.4
_O_RI_C = 0.3
_O_CV = 1.6
_O_NEG_BETA_T = 0.2
_O_ZETA_M = 0.2
_O_ZETA_S = 1.0
_O_A_M = 1.26
_O_C_M = 8.38
_O_A_S = -28.86
_O_C_S = 98.96
_O_C16 = 16.0
_O_C5 = 5.0
_O_EPS_LMD = 0.1


def _wm_ws_oracle(ustar, Bf, d, h, eps):
    """Independent LMD94 App. B similarity velocity scales (plain Python)."""
    d_eff = min(d, _O_EPS_LMD * h)
    Bf_safe = Bf if abs(Bf) > eps else math.copysign(eps, Bf)
    L_MO = ustar ** 3 / (_O_KAPPA * Bf_safe)
    zeta = d_eff / L_MO
    abs_zeta = abs(zeta)
    Bf_pos = max(Bf, 0.0)
    base16 = max(1.0 + _O_C16 * abs_zeta, 1.0)
    wm_weak = _O_KAPPA * ustar * base16 ** 0.25
    ws_weak = _O_KAPPA * ustar * base16 ** 0.5
    m_base = max(_O_A_M * ustar**3 + _O_C_M * _O_KAPPA * Bf_pos * d_eff, 1e-30)
    s_base = max(_O_A_S * ustar**3 + _O_C_S * _O_KAPPA * Bf_pos * d_eff, 1e-30)
    wm_conv = _O_KAPPA * m_base ** (1.0 / 3.0)
    ws_conv = _O_KAPPA * s_base ** (1.0 / 3.0)
    wm_unstable = wm_weak if abs_zeta <= _O_ZETA_M else wm_conv
    ws_unstable = ws_weak if abs_zeta <= _O_ZETA_S else ws_conv
    w_stable = _O_KAPPA * ustar / max(1.0 + _O_C5 * max(-zeta, 0.0), 1.0)
    is_unstable = Bf > 0.0
    wm = max(wm_unstable if is_unstable else w_stable, 1e-10)
    ws = max(ws_unstable if is_unstable else w_stable, 1e-10)
    return wm, ws


def _vt2_oracle(N, d, w_s):
    return (_O_CV * _O_NEG_BETA_T ** 0.5 * N * d * w_s
            / (_O_RI_C * _O_KAPPA ** 2 * math.sqrt(_O_C_S * _O_EPS_LMD)))


# (u_star [m/s], B_f [m^2/s^3], h_bl [m]).  Signs: B_f > 0 destabilising.
_STATES = [
    (0.010, -1.0e-6, 100.0),   # stable (B_f < 0)
    (0.010, +1.0e-7, 100.0),   # weakly unstable (small B_f -> large L -> small zeta)
    (0.005, +1.0e-6, 100.0),   # convective (deep d -> large zeta)
    (0.008, +3.0e-7, 80.0),    # mixed: crosses zeta_m and zeta_s as d grows
]
_DEPTHS = [0.5, 2.0, 5.0, 8.0, 12.0, 30.0, 60.0, 100.0]   # incl. d > 0.1*h (cap)
_CFG = KPPConfig()


def _prod_wm_ws(ustar, Bf, d, h):
    wm, ws = _kpp_velocity_scales(
        jnp.array([ustar]), jnp.array([Bf]),
        jnp.array([[d]]), jnp.array([[h]]), _CFG, _EPS)
    return float(wm[0, 0]), float(ws[0, 0])


# --- velocity scales --------------------------------------------------------

@pytest.mark.parametrize("ustar,Bf,h", _STATES)
@pytest.mark.parametrize("d", _DEPTHS)
def test_velocity_scales_match_lmd94(ustar, Bf, h, d):
    wm, ws = _prod_wm_ws(ustar, Bf, d, h)
    wm_o, ws_o = _wm_ws_oracle(ustar, Bf, d, h, _EPS)
    assert wm == pytest.approx(wm_o, rel=1e-12, abs=0.0)
    assert ws == pytest.approx(ws_o, rel=1e-12, abs=0.0)


def test_velocity_scales_exercise_all_regimes():
    """Non-vacuity: the grid must actually visit stable, weakly-unstable,
    convective, AND the split regime where w_m is convective (|zeta| > zeta_m)
    while w_s is still weak (|zeta| < zeta_s)."""
    saw_stable = saw_weak = saw_conv = saw_split = False
    for ustar, Bf, h in _STATES:
        for d in _DEPTHS:
            d_eff = min(d, _O_EPS_LMD * h)
            if Bf <= 0.0:
                saw_stable = True
                continue
            zeta = d_eff / (ustar**3 / (_O_KAPPA * Bf))
            if zeta <= _O_ZETA_M:
                saw_weak = True
            if zeta > _O_ZETA_S:
                saw_conv = True
            if _O_ZETA_M < zeta <= _O_ZETA_S:
                saw_split = True
    assert saw_stable and saw_weak and saw_conv and saw_split


def _ws_weak(ustar, zeta):
    return _O_KAPPA * ustar * max(1.0 + _O_C16 * abs(zeta), 1.0) ** 0.5


def _wm_weak(ustar, zeta):
    return _O_KAPPA * ustar * max(1.0 + _O_C16 * abs(zeta), 1.0) ** 0.25


def _ws_conv(ustar, Bf, d_eff):
    base = max(_O_A_S * ustar**3 + _O_C_S * _O_KAPPA * max(Bf, 0.0) * d_eff, 1e-30)
    return _O_KAPPA * base ** (1.0 / 3.0)


def _wm_conv(ustar, Bf, d_eff):
    base = max(_O_A_M * ustar**3 + _O_C_M * _O_KAPPA * max(Bf, 0.0) * d_eff, 1e-30)
    return _O_KAPPA * base ** (1.0 / 3.0)


def test_regime_joins_constructed_and_pinned():
    """CONSTRUCTED cases landing at exact zeta values that straddle BOTH joins
    (momentum zeta_m = 0.2, scalar zeta_s = 1.0), so the inclusive `<=` switch is
    directly pinned — not just spanned by the grid.  For each: production == the
    independent oracle to round-off, AND the branch identity is asserted
    non-vacuously (the weak and convective forms genuinely differ there):
      - zeta < 0.2:  w_m weak,  w_s weak
      - 0.2 < zeta < 1.0 (split): w_m CONVECTIVE, w_s still weak
      - zeta > 1.0:  w_m conv,   w_s CONVECTIVE
    zeta = d_eff*kappa*B_f/u*^3 with d_eff = d < epsilon*h.  (u*, d) are chosen so
    the B_f needed for each target zeta stays WELL ABOVE _EPS, otherwise the
    near-zero copysign(eps, B_f) guard would floor B_f and inflate the effective
    zeta out of the intended branch.)"""
    ustar, d, h = 0.02, 2.0, 100.0                  # d_eff = d = 2 (< 0.1*h = 10)

    def bf_for(zeta):
        # zeta * u*^3 / (d*kappa) = zeta * 1e-5 here, so B_f >= 1.5e-6 >> _EPS.
        return zeta * ustar**3 / (d * _O_KAPPA)

    for zeta in (0.15, 0.2, 0.5, 0.9, 1.0, 1.1):
        Bf = bf_for(zeta)
        wm, ws = _prod_wm_ws(ustar, Bf, d, h)
        wm_o, ws_o = _wm_ws_oracle(ustar, Bf, d, h, _EPS)
        assert wm == pytest.approx(wm_o, rel=1e-12), f"w_m at zeta={zeta}"
        assert ws == pytest.approx(ws_o, rel=1e-12), f"w_s at zeta={zeta}"

    # Momentum branch identity across zeta_m = 0.2 (non-vacuous: weak != conv).
    wm_lo = _prod_wm_ws(ustar, bf_for(0.15), d, h)[0]
    wm_hi = _prod_wm_ws(ustar, bf_for(0.30), d, h)[0]
    assert wm_lo == pytest.approx(_wm_weak(ustar, 0.15), rel=1e-12)
    assert wm_hi == pytest.approx(_wm_conv(ustar, bf_for(0.30), d), rel=1e-12)
    assert _wm_weak(ustar, 0.30) != pytest.approx(_wm_conv(ustar, bf_for(0.30), d), rel=1e-6)

    # Scalar branch identity across zeta_s = 1.0 (w_s still weak in the split).
    ws_lo = _prod_wm_ws(ustar, bf_for(0.5), d, h)[1]     # split: w_s still weak
    ws_hi = _prod_wm_ws(ustar, bf_for(1.1), d, h)[1]     # w_s now convective
    assert ws_lo == pytest.approx(_ws_weak(ustar, 0.5), rel=1e-12)
    assert ws_hi == pytest.approx(_ws_conv(ustar, bf_for(1.1), d), rel=1e-12)
    assert _ws_weak(ustar, 1.1) != pytest.approx(_ws_conv(ustar, bf_for(1.1), d), rel=1e-6)


def test_surface_layer_cap_holds_scale_constant():
    """DEPARTURE canary: below sigma = epsilon (d > epsilon*h) the similarity
    scale is capped at d_eff = epsilon*h, so w is identical at two deeper d."""
    ustar, Bf, h = 0.005, 1.0e-6, 100.0     # 0.1*h = 10 m
    w1 = _prod_wm_ws(ustar, Bf, 40.0, h)
    w2 = _prod_wm_ws(ustar, Bf, 90.0, h)
    assert w1[0] == pytest.approx(w2[0], rel=1e-12)
    assert w1[1] == pytest.approx(w2[1], rel=1e-12)


def test_scalar_exceeds_momentum_weakly_unstable():
    """In the weakly-unstable branch w_s uses the steeper exponent (Pr_t < 1):
    w_s = kappa*u*(1+16|zeta|)^{1/2} > w_m = kappa*u*(1+16|zeta|)^{1/4}."""
    wm, ws = _prod_wm_ws(0.01, 1.0e-7, 5.0, 100.0)   # small zeta, weakly-unstable
    assert ws > wm


def test_near_zero_buoyancy_copysign_branch():
    """|B_f| <= eps takes the copysign(eps, B_f) guard (issue #168); production
    must match the oracle there too (finite, positive scales)."""
    ustar, h = 0.01, 100.0
    for Bf in (0.5 * _EPS, -0.5 * _EPS):
        wm, ws = _prod_wm_ws(ustar, Bf, 5.0, h)
        wm_o, ws_o = _wm_ws_oracle(ustar, Bf, 5.0, h, _EPS)
        assert wm == pytest.approx(wm_o, rel=1e-12)
        assert ws == pytest.approx(ws_o, rel=1e-12)


# --- V_t^2 (LMD94 Eq. 23) ---------------------------------------------------

@pytest.mark.parametrize("N,d,w_s", [(0.01, 20.0, 0.02), (0.005, 5.0, 0.01),
                                     (0.02, 50.0, 0.03)])
def test_vt2_matches_lmd94_eq23(N, d, w_s):
    got = float(_kpp_unresolved_shear_variance(
        jnp.array(N), jnp.array(d), jnp.array(w_s), _CFG, _EPS))
    assert got == pytest.approx(_vt2_oracle(N, d, w_s), rel=1e-12, abs=0.0)


def test_vt2_trilinear_and_prefactor():
    """V_t^2 is linear in each of N, d, w_s; and its prefactor is the LMD94
    Cv*sqrt(-beta_T)/(Ri_c*kappa^2*sqrt(c_s*eps)) (not 2.236x larger)."""
    base = float(_kpp_unresolved_shear_variance(
        jnp.array(0.01), jnp.array(10.0), jnp.array(0.02), _CFG, _EPS))
    doubled = float(_kpp_unresolved_shear_variance(
        jnp.array(0.02), jnp.array(10.0), jnp.array(0.02), _CFG, _EPS))
    assert doubled / base == pytest.approx(2.0, rel=1e-12)
    prefactor = base / (0.01 * 10.0 * 0.02)
    expected = _O_CV * _O_NEG_BETA_T**0.5 / (_O_RI_C * _O_KAPPA**2 * math.sqrt(_O_C_S * _O_EPS_LMD))
    assert prefactor == pytest.approx(expected, rel=1e-12)


# --- shape function G(sigma) ------------------------------------------------

@pytest.mark.parametrize("sigma", [-0.3, 0.0, 0.1, 1.0 / 3.0, 0.5, 0.9, 1.0, 1.4])
def test_shape_function_matches_cubic(sigma):
    sc = min(max(sigma, 0.0), 1.0)
    assert float(_kpp_shape_function(jnp.array(sigma))) == pytest.approx(
        sc * (1.0 - sc) ** 2, rel=1e-12, abs=0.0)


def test_shape_function_endpoints_and_peak():
    assert float(_kpp_shape_function(jnp.array(0.0))) == 0.0
    assert float(_kpp_shape_function(jnp.array(1.0))) == 0.0
    assert float(_kpp_shape_function(jnp.array(1.0 / 3.0))) == pytest.approx(4.0 / 27.0, rel=1e-12)


def test_shape_function_base_derivative_zero_is_the_departure():
    """DEPARTURE canary: G(1) = G'(1) = 0 for the reduced cubic sigma*(1-sigma)^2,
    so BOTH K_bl and dK_bl/dz vanish at the BL base.  The reduced form thus cannot
    represent a nonzero interior-matched base value/slope (the LMD94 App. B / Eq. D
    matching cubic sets a2, a3 to match the interior K and its derivative at
    sigma=1); this is the documented under-entrainment gap.  G'(1/3)=0 is the
    interior peak."""
    dG = jax.grad(lambda s: _kpp_shape_function(s))
    assert float(_kpp_shape_function(jnp.array(1.0))) == 0.0
    assert float(dG(jnp.array(1.0))) == pytest.approx(0.0, abs=1e-12)
    # and it is genuinely a cubic bump: G'(1/3) = 0 at the interior peak
    assert float(dG(jnp.array(1.0 / 3.0))) == pytest.approx(0.0, abs=1e-12)


# --- coefficient canary -----------------------------------------------------

def test_kpp_config_matches_lmd94_appendix_b():
    c = KPPConfig()
    assert c.kappa_vk == _O_KAPPA
    assert c.Ri_crit == _O_RI_C
    assert c.Cv == _O_CV
    assert c.neg_beta_T == _O_NEG_BETA_T
    assert (c.zeta_m_abs, c.zeta_s_abs) == (_O_ZETA_M, _O_ZETA_S)
    assert (c.a_m, c.c_m, c.a_s, c.c_s) == (_O_A_M, _O_C_M, _O_A_S, _O_C_S)
    assert (c.businger_unstable_coeff, c.businger_stable_coeff) == (_O_C16, _O_C5)
    assert c.epsilon_lmd == _O_EPS_LMD


# --- AD-safety --------------------------------------------------------------

def test_kpp_forms_grads_finite():
    """grads of w_m (wrt u*), V_t^2 (wrt N), and G (wrt sigma) are finite."""
    def _wm(us):
        wm, _ = _kpp_velocity_scales(us[None], jnp.array([1.0e-6]),
                                     jnp.array([[5.0]]), jnp.array([[100.0]]), _CFG, _EPS)
        return wm[0, 0]
    g1 = jax.grad(_wm)(jnp.array(0.01))
    assert bool(jnp.isfinite(g1)) and float(jnp.abs(g1)) > 0.0

    g2 = jax.grad(lambda N: _kpp_unresolved_shear_variance(
        N, jnp.array(10.0), jnp.array(0.02), _CFG, _EPS))(jnp.array(0.01))
    assert bool(jnp.isfinite(g2)) and float(jnp.abs(g2)) > 0.0

    g3 = jax.grad(lambda s: _kpp_shape_function(s))(jnp.array(0.5))
    assert bool(jnp.isfinite(g3))
