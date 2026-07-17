"""Faithfulness pins for the Holland-Jenkins three-equation ice-shelf melt.

Target: ``freezing_point_C`` and ``three_equation_melt`` (+ the
``compute_basal_melt`` dispatcher) in ``legoesm.ocean.physics.ice_shelf`` — the
closed-form basal-melt parameterisation for an ice-shelf cavity.

Most-trustful source
--------------------
Holland & Jenkins (1999), "Modeling thermodynamic ice-ocean interactions at the
base of an ice shelf", JPO 29, 1787-1800; freezing-point relation Jenkins (1991).
Three governing equations at the ice-ocean interface (zero ice salinity, no ice
conduction):

  heat     :  rho_w * c_w * gamma_T * (T_amb - T_b)  =  rho_i * L_f * m_dot
  salt     :  rho_w * gamma_S       * (S_amb - S_b)  =  rho_i * m_dot * S_b
  liquidus :  T_b = a*S_b + b + c*p

Eliminating T_b, S_b gives a quadratic A*m^2 + B*m + C = 0; the physical root is
m_dot = (-B + sqrt(B^2-4AC)) / (2A).

The existing test_ice_shelf.py is behavioral (0 exact-magnitude assertions); this
pins the liquidus coefficients, the quadratic coefficients + physical root, the
small-melt Beckmann-Goosse limit, and — as a reimplementation-independent truth
tier — that the closed-form solution satisfies all THREE governing equations.

Certification (test-only):
1. Liquidus T_f = a*S + b + c*p exact + the Jenkins (1991) coefficients.
2. Quadratic A/B/C independent construction -> m_dot; physical (+sqrt) root; the
   correct small-theta asymptotic m_dot -> -C/B (NOT the docstring's
   alpha*theta/gamma Beckmann-Goosse form, which needs gamma*beta >>
   alpha*delta*(T-b-cp) -- false at the default params).
3. TRUTH TIER: at the solution, heat (Q == rho_i*L_f*m_dot), salt, and liquidus
   all hold simultaneously (the quadratic solve is self-consistent).
4. Output signs (melt vs freeze-on), dispatch raise / disabled no-op, coefficient
   plumbing, differentiability (incl. the freeze-on disc<=0 sqrt edge).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.ocean.physics.ice_shelf import (
    IceShelfConfig,
    compute_basal_melt,
    freezing_point_C,
    three_equation_melt,
)

jax.config.update("jax_enable_x64", True)

_CFG = IceShelfConfig(enabled=True, scheme="three_equation")


def _greek(cfg):
    """The four lumped coefficients of the three-equation quadratic."""
    alpha = cfg.rho_w * cfg.c_w * cfg.gamma_T      # W/(m^2 K)
    beta = cfg.rho_w * cfg.gamma_S                 # kg/(m^2 s)
    gamma = cfg.rho_ice * cfg.L_f                  # J/m^3
    delta = cfg.rho_ice                            # kg/m^3
    return alpha, beta, gamma, delta


def _melt(T, S, p, cfg=_CFG):
    return three_equation_melt(jnp.asarray(float(T)), jnp.asarray(float(S)),
                               jnp.asarray(float(p)), config=cfg)


# ---------------------------------------------------------------------------
# 1. Freezing-point liquidus (Jenkins 1991).
# ---------------------------------------------------------------------------
def test_freezing_point_liquidus_exact():
    S, p = 34.5, 500.0
    tf = float(freezing_point_C(jnp.asarray(S), jnp.asarray(p), config=_CFG))
    # Independent Jenkins (1991) linear liquidus with literal coefficients.
    np.testing.assert_allclose(tf, -5.73e-2 * S + 8.32e-2 + -7.61e-4 * p, rtol=1e-12)


def test_freezing_point_coefficients():
    assert _CFG.freeze_a == -5.73e-2      # degC / PSU
    assert _CFG.freeze_b == 8.32e-2       # degC
    assert _CFG.freeze_c == -7.61e-4      # degC / dbar


def test_freezing_point_pressure_and_salinity_dependence():
    # deeper (higher p) and saltier both LOWER the freezing point.
    base = float(freezing_point_C(jnp.asarray(34.0), jnp.asarray(0.0), config=_CFG))
    deep = float(freezing_point_C(jnp.asarray(34.0), jnp.asarray(1000.0), config=_CFG))
    salty = float(freezing_point_C(jnp.asarray(36.0), jnp.asarray(0.0), config=_CFG))
    np.testing.assert_allclose(deep - base, -7.61e-4 * 1000.0, rtol=1e-12)
    np.testing.assert_allclose(salty - base, -5.73e-2 * 2.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. Three-equation quadratic + root.
# ---------------------------------------------------------------------------
def test_quadratic_coefficients_and_physical_root():
    T, S, p = 1.2, 34.7, 600.0            # warm ambient -> melt
    r = _melt(T, S, p)
    a, b, g, d = _greek(_CFG)
    tf_amb = -5.73e-2 * S + 8.32e-2 + -7.61e-4 * p
    theta = T - tf_amb
    t_minus_bcp = T - 8.32e-2 - (-7.61e-4) * p
    A = g * d
    B = g * b - a * d * t_minus_bcp
    C = -a * b * theta
    disc = B * B - 4.0 * A * C
    m_expected = (-B + np.sqrt(disc)) / (2.0 * A)   # physical (+sqrt) root
    np.testing.assert_allclose(float(r.m_dot_m_s), m_expected, rtol=1e-12)
    assert float(r.m_dot_m_s) > 0.0                  # warm ambient melts


def test_small_theta_first_order_root():
    # For weak thermal driving the physical root -> -C/B (first-order Taylor of
    # (-B + sqrt(B^2-4AC))/(2A) as C -> 0).  NOTE: the module docstring's
    # "reduces to alpha*theta/gamma (Beckmann-Goosse)" reduction only holds when
    # gamma*beta >> alpha*delta*(T-b-cp); at the DEFAULT params B is dominated by
    # the alpha*delta*(T-b-cp) term (T-b-cp ~ a_liq*S ~ -2, so gamma_T largely
    # CANCELS), so the correct small-melt asymptotic is -C/B, pinned here.
    S, p = 34.5, 400.0
    theta = 1.0e-4
    tf_amb = -5.73e-2 * S + 8.32e-2 + -7.61e-4 * p
    r = _melt(tf_amb + theta, S, p)
    a, b, g, d = _greek(_CFG)
    t_minus_bcp = (tf_amb + theta) - 8.32e-2 - (-7.61e-4) * p
    B = g * b - a * d * t_minus_bcp
    C = -a * b * theta
    np.testing.assert_allclose(float(r.m_dot_m_s), -C / B, rtol=1e-3)   # first-order root


def test_three_equation_governing_equations_hold():
    # TRUTH TIER (reimplementation-independent): the closed-form solution must
    # satisfy all three governing equations simultaneously.  Salt and liquidus
    # hold by construction of S_b, T_b; the HEAT balance validates the quadratic.
    T, S, p = 1.5, 34.8, 700.0
    r = _melt(T, S, p)
    a, b, g, d = _greek(_CFG)
    m, T_b, S_b = float(r.m_dot_m_s), float(r.T_b_C), float(r.S_b_PSU)
    # heat:  alpha*(T_amb - T_b) == gamma*m   (== rho_i*L_f*m)
    np.testing.assert_allclose(a * (T - T_b), g * m, rtol=1e-10)
    np.testing.assert_allclose(float(r.heat_extracted_from_ocean), g * m, rtol=1e-10)
    # salt:  beta*(S_amb - S_b) == delta*m*S_b
    np.testing.assert_allclose(b * (S - S_b), d * m * S_b, rtol=1e-10)
    # liquidus: T_b == a_liq*S_b + b_liq + c_liq*p
    np.testing.assert_allclose(T_b, -5.73e-2 * S_b + 8.32e-2 + -7.61e-4 * p, rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. Interface T_b / S_b + outputs.
# ---------------------------------------------------------------------------
def test_interface_salinity_balance():
    T, S, p = 1.0, 34.6, 500.0
    r = _melt(T, S, p)
    a, b, g, d = _greek(_CFG)
    m = float(r.m_dot_m_s)
    np.testing.assert_allclose(float(r.S_b_PSU), b * S / (d * m + b), rtol=1e-10)


def test_freshwater_and_heat_outputs():
    T, S, p = 1.3, 34.7, 550.0
    r = _melt(T, S, p)
    a, _, _, _ = _greek(_CFG)
    # freshwater INTO ocean = rho_ice * m_dot; melt (>0) adds freshwater.
    np.testing.assert_allclose(float(r.freshwater_to_ocean), _CFG.rho_ice * float(r.m_dot_m_s),
                               rtol=1e-12)
    np.testing.assert_allclose(float(r.heat_extracted_from_ocean),
                               a * (T - float(r.T_b_C)), rtol=1e-12)
    assert float(r.freshwater_to_ocean) > 0.0 and float(r.heat_extracted_from_ocean) > 0.0


def test_freeze_on_negative_melt():
    # Ambient BELOW the in-situ freezing point -> freeze-on (m_dot < 0).
    p, S = 500.0, 34.5
    tf = -5.73e-2 * S + 8.32e-2 + -7.61e-4 * p
    r = _melt(tf - 0.05, S, p)            # 0.05 K supercooled
    assert float(r.m_dot_m_s) < 0.0


# ---------------------------------------------------------------------------
# 4. Dispatch, disabled, plumbing, differentiability.
# ---------------------------------------------------------------------------
def test_dispatch_raises_on_unknown_scheme():
    bad = IceShelfConfig(enabled=True, scheme="quadratic")
    try:
        compute_basal_melt(jnp.asarray(1.0), jnp.asarray(34.5), jnp.asarray(500.0), config=bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError for scheme='quadratic'")


def test_disabled_returns_zero_melt():
    off = IceShelfConfig(enabled=False)
    r = compute_basal_melt(jnp.asarray(1.0), jnp.asarray(34.5), jnp.asarray(500.0), config=off)
    assert float(r.m_dot_m_s) == 0.0
    assert float(r.freshwater_to_ocean) == 0.0
    assert float(r.heat_extracted_from_ocean) == 0.0
    # interface T reported as the ambient freezing point.
    np.testing.assert_allclose(float(r.T_b_C),
                               -5.73e-2 * 34.5 + 8.32e-2 + -7.61e-4 * 500.0, rtol=1e-12)


def test_gamma_t_plumbing_flows_through_quadratic():
    # Non-default gamma_T flows through to m via the full quadratic (NOT simply
    # proportional to gamma_T -- see the first-order-root note above -- so verify
    # against the recomputed quadratic root, and that it differs from the default).
    T, S, p = 1.2, 34.7, 600.0
    cfg2 = _CFG._replace(gamma_T=2.0 * _CFG.gamma_T)
    r = _melt(T, S, p, cfg2)
    a, b, g, d = _greek(cfg2)                     # alpha uses the doubled gamma_T
    tf_amb = -5.73e-2 * S + 8.32e-2 + -7.61e-4 * p
    theta = T - tf_amb
    t_minus_bcp = T - 8.32e-2 - (-7.61e-4) * p
    A = g * d
    B = g * b - a * d * t_minus_bcp
    C = -a * b * theta
    m_exp = (-B + np.sqrt(B * B - 4.0 * A * C)) / (2.0 * A)
    np.testing.assert_allclose(float(r.m_dot_m_s), m_exp, rtol=1e-12)
    assert abs(float(r.m_dot_m_s) - float(_melt(T, S, p, _CFG).m_dot_m_s)) > 1e-12


def test_freeze_coefficient_plumbing():
    # A different liquidus slope changes the freezing point AND flows through to
    # the melt rate (a melt impl hardcoding the default liquidus coeffs fails the
    # second check).
    S, p = 34.5, 500.0
    base = float(freezing_point_C(jnp.asarray(S), jnp.asarray(p), config=_CFG))
    cfg_a = _CFG._replace(freeze_a=-6.0e-2)
    steeper = float(freezing_point_C(jnp.asarray(S), jnp.asarray(p), config=cfg_a))
    np.testing.assert_allclose(steeper - base, (-6.0e-2 - (-5.73e-2)) * S, rtol=1e-10)
    # freeze_a flows into three_equation_melt: verify m against the recomputed
    # quadratic with the new coefficient, and that it differs from the default.
    T = 1.2
    r = _melt(T, S, p, cfg_a)
    a, b, g, d = _greek(cfg_a)
    tf_amb = -6.0e-2 * S + 8.32e-2 + -7.61e-4 * p           # uses the NEW freeze_a
    theta = T - tf_amb
    t_minus_bcp = T - 8.32e-2 - (-7.61e-4) * p
    m_exp = (-(g * b - a * d * t_minus_bcp)
             + np.sqrt((g * b - a * d * t_minus_bcp) ** 2
                       - 4.0 * (g * d) * (-a * b * theta))) / (2.0 * (g * d))
    np.testing.assert_allclose(float(r.m_dot_m_s), m_exp, rtol=1e-12)
    assert abs(float(r.m_dot_m_s) - float(_melt(T, S, p, _CFG).m_dot_m_s)) > 1e-12


def test_gamma_s_plumbing_flows_through_quadratic():
    # gamma_S enters beta = rho_w*gamma_S; a non-default value flows through the
    # quadratic to m (and to the interface S_b), differing from the default.
    T, S, p = 1.2, 34.7, 600.0
    cfg2 = _CFG._replace(gamma_S=2.0 * _CFG.gamma_S)
    r = _melt(T, S, p, cfg2)
    a, b, g, d = _greek(cfg2)                     # beta uses the doubled gamma_S
    tf_amb = -5.73e-2 * S + 8.32e-2 + -7.61e-4 * p
    theta = T - tf_amb
    t_minus_bcp = T - 8.32e-2 - (-7.61e-4) * p
    m_exp = (-(g * b - a * d * t_minus_bcp)
             + np.sqrt((g * b - a * d * t_minus_bcp) ** 2
                       - 4.0 * (g * d) * (-a * b * theta))) / (2.0 * (g * d))
    np.testing.assert_allclose(float(r.m_dot_m_s), m_exp, rtol=1e-12)
    assert abs(float(r.m_dot_m_s) - float(_melt(T, S, p, _CFG).m_dot_m_s)) > 1e-12


def test_dispatcher_routes_three_equation():
    # compute_basal_melt(scheme='three_equation') must route to three_equation_melt.
    T, S, p = 1.2, 34.7, 600.0
    via_dispatch = compute_basal_melt(jnp.asarray(T), jnp.asarray(S), jnp.asarray(p),
                                      config=_CFG)
    direct = _melt(T, S, p, _CFG)
    np.testing.assert_allclose(float(via_dispatch.m_dot_m_s), float(direct.m_dot_m_s),
                               rtol=1e-12)
    np.testing.assert_allclose(float(via_dispatch.S_b_PSU), float(direct.S_b_PSU), rtol=1e-12)


def test_differentiable_ordinary_branch():
    def loss(T):
        r = three_equation_melt(T, jnp.asarray(34.6), jnp.asarray(500.0), config=_CFG)
        return r.m_dot_m_s
    g_melt = jax.grad(loss)(jnp.asarray(1.0))          # warm (disc > 0)
    assert jnp.isfinite(g_melt)


def test_differentiable_at_disc_le_zero_guard():
    # The double-where sqrt(disc) must keep the gradient finite when disc <= 0.
    # For the DEFAULT params disc = B^2 + 4*gamma*delta*alpha*beta*theta stays > 0
    # for all physical theta (B grows quadratically), so exercise the guard with a
    # pathological out-of-range input (negative "salinity", as a halo/garbage cell
    # would produce): S=-34.9, p=500, T=-0.30 -> disc < 0.
    a, b, g, d = _greek(_CFG)
    S_bad, p, T = -34.9, 500.0, -0.30
    tf_amb = -5.73e-2 * S_bad + 8.32e-2 + -7.61e-4 * p
    theta = T - tf_amb
    t_minus_bcp = T - 8.32e-2 - (-7.61e-4) * p
    disc = (g * b - a * d * t_minus_bcp) ** 2 - 4.0 * (g * d) * (-a * b * theta)
    assert disc < 0.0                                   # precondition: guard branch is live

    def loss(T_in):
        r = three_equation_melt(T_in, jnp.asarray(S_bad), jnp.asarray(p), config=_CFG)
        return r.m_dot_m_s
    g_edge = jax.grad(loss)(jnp.asarray(T))
    assert jnp.isfinite(g_edge)
