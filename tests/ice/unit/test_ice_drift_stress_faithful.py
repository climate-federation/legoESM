"""Faithfulness pins for the sea-ice momentum FORCING closed forms.

Target: the drag + free-drift closed forms in ``legoesm.ice.dynamics`` —
``air_ice_stress``, ``ocean_ice_stress`` (quadratic bulk drag) and
``free_drift_velocity`` (the Nansen-Zubov steady free-drift limit).  These are
the momentum-forcing terms of the EVP momentum equation, distinct from the VP
internal-stress rheology (already pinned by test_ice_rheology_faithful.py) and
the iteration-coupled EVP subcycle (evp_solver / mevp_solver).

Most-trustful source
--------------------
* Quadratic bulk drag (Hunke & Dukowicz 1997, eq. for the air/ocean stress):
      tau = rho * C_d * |U_rel| * U_rel      [component-wise, U_rel a 2-vector]
  legoESM uses the ZERO-turning-angle form (a documented departure from the
  full HD97 stress, which rotates U_rel by an air/water turning angle).  A
  1e-10 (m/s)^2 term is ADDED under the square root (an effective ~1e-5 m/s
  speed floor, NOT a 1e-10 m/s floor) for a finite adjoint at U_rel = 0.
* Free drift (Nansen 1902 / Zubov; the classic 2 % rule):
      U_ice = U_ocean + alpha * (U_wind - U_ocean),
      alpha = sqrt(rho_air * C_ai / (rho_ocean * C_oi)),
  the steady no-Coriolis no-internal-stress balance under the wind>>ice
  approximation (the air relative velocity is taken as U_wind - U_ocean).  With
  the CICE-default drags this is alpha = 0.016807 (ice drifts at ~1.68 % of the
  wind relative to the ocean).

Certification (test-only; the module is a set of closed forms):
1. EXACT quadratic-drag form (tau == rho*C*sqrt(du^2+dv^2+1e-10)*du) pinned to a
   tight rtol against an independent reimplementation typed from the formula.
2. TRUTH-TIER / physical properties: quadratic scaling (2x U_rel -> 4x |tau|),
   stress PARALLEL to U_rel (no turning angle — the HD97 departure), the ~1e-5 m/s
   effective floor VALUE pinned near zero, and the free-drift Nansen balance pinned
   to its EXACT analytic residual (2a-a^2)/(1-a)^2.
3. free_drift alpha == sqrt(rho_air*C_ai/(rho_oc*C_oi)) + the ~1.68 % magnitude.
4. FULL coefficient plumbing (BOTH rho AND C_d varied, for BOTH stress fns AND
   free_drift densities); bound signature defaults proven == config/constants via
   omitted-argument calls; differentiability at the floor; config defaults.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.dynamics import (
    air_ice_stress,
    free_drift_velocity,
    ocean_ice_stress,
)

from legoesm import constants

jax.config.update("jax_enable_x64", True)  # matches the sibling ice faithful tests

_CFG = SeaIceConfig()
_RHO_AIR = float(constants.rho_air)       # 1.225
_RHO_OCEAN = float(constants.rho_ocean)   # 1025.0
_C_AI = _CFG.drag_atm                      # 1.3e-3
_C_OI = _CFG.drag_ocean                    # 5.5e-3
_FLOOR_SQ = 1e-10                          # added under the sqrt: (m/s)^2
_FLOOR_SPEED = np.sqrt(_FLOOR_SQ)          # effective ~1e-5 m/s speed floor


def _arr(x):
    return jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64))


def _f(x):
    return float(x[0])


def _drag_oracle(u_i, v_i, u_ref, v_ref, rho, cd):
    """Independent quadratic-drag reimplementation (typed from the formula)."""
    du, dv = u_ref - u_i, v_ref - v_i
    speed = np.sqrt(du ** 2 + dv ** 2 + _FLOOR_SQ)
    return rho * cd * speed * du, rho * cd * speed * dv


# ---------------------------------------------------------------------------
# 1. EXACT quadratic-drag form (tight coefficient pin vs independent oracle).
# ---------------------------------------------------------------------------
def test_air_ice_stress_exact_form():
    u_i, v_i, u_w, v_w = 0.1, -0.05, 8.0, -3.0
    tx, ty = air_ice_stress(_arr(u_i), _arr(v_i), _arr(u_w), _arr(v_w),
                            _RHO_AIR, _C_AI)
    ex, ey = _drag_oracle(u_i, v_i, u_w, v_w, _RHO_AIR, _C_AI)
    np.testing.assert_allclose(_f(tx), ex, rtol=1e-12)
    np.testing.assert_allclose(_f(ty), ey, rtol=1e-12)


def test_ocean_ice_stress_exact_form():
    u_i, v_i, u_o, v_o = 0.2, 0.1, 0.4, -0.15
    tx, ty = ocean_ice_stress(_arr(u_i), _arr(v_i), _arr(u_o), _arr(v_o),
                              _RHO_OCEAN, _C_OI)
    ex, ey = _drag_oracle(u_i, v_i, u_o, v_o, _RHO_OCEAN, _C_OI)
    np.testing.assert_allclose(_f(tx), ex, rtol=1e-12)
    np.testing.assert_allclose(_f(ty), ey, rtol=1e-12)


def test_drag_quadratic_scaling():
    # |tau| ~ |U_rel|^2: doubling the relative velocity quadruples the magnitude
    # (a linear drag would only double it).
    tx1, ty1 = air_ice_stress(_arr(0.0), _arr(0.0), _arr(3.0), _arr(4.0),
                              _RHO_AIR, _C_AI)               # |U_rel| = 5
    tx2, ty2 = air_ice_stress(_arr(0.0), _arr(0.0), _arr(6.0), _arr(8.0),
                              _RHO_AIR, _C_AI)               # |U_rel| = 10
    mag1 = np.hypot(_f(tx1), _f(ty1))
    mag2 = np.hypot(_f(tx2), _f(ty2))
    np.testing.assert_allclose(mag2 / mag1, 4.0, rtol=1e-9)


def test_drag_parallel_no_turning_angle():
    # DEPARTURE canary: legoESM's stress is PARALLEL to U_rel (tau_x/tau_y ==
    # du/dv) — the full HD97 stress rotates U_rel by a turning angle, which would
    # break this ratio.
    du, dv = 5.0, -2.0
    tx, ty = air_ice_stress(_arr(0.0), _arr(0.0), _arr(du), _arr(dv),
                            _RHO_AIR, _C_AI)
    np.testing.assert_allclose(_f(tx) / _f(ty), du / dv, rtol=1e-12)


def test_drag_zero_relative_velocity_is_zero():
    # At U_rel = 0 exactly the stress is exactly zero: tau = rho*C*speed*du with
    # du = 0 kills the product regardless of the floored speed (= 1e-5 m/s).
    tx, ty = ocean_ice_stress(_arr(0.3), _arr(0.3), _arr(0.3), _arr(0.3),
                              _RHO_OCEAN, _C_OI)
    assert _f(tx) == 0.0 and _f(ty) == 0.0


def test_drag_speed_floor_value():
    # Pin the FLOOR MAGNITUDE independently for BOTH stress fns (each has its own
    # +1e-10 literal): at |U_rel| = 1e-8 << 1e-5 the added 1e-10 (m/s)^2 dominates,
    # so the effective speed is sqrt(1e-10 + du^2) ≈ 1e-5 m/s.  tau_x / (rho*C*du)
    # recovers that speed — distinguishing the 1e-10 floor from a wrong one (a 1e-8
    # floor would give 1e-4, no floor would give 1e-8).
    du = 1e-8
    for stress, rho, cd in ((air_ice_stress, _RHO_AIR, _C_AI),
                            (ocean_ice_stress, _RHO_OCEAN, _C_OI)):
        tx, _ = stress(_arr(0.0), _arr(0.0), _arr(du), _arr(0.0), rho, cd)
        implied_speed = _f(tx) / (rho * cd * du)
        np.testing.assert_allclose(implied_speed, np.sqrt(_FLOOR_SQ + du ** 2), rtol=1e-9)


def test_air_stress_coefficient_plumbing():
    # Independent linear dependence on EACH of rho_air and C_ai (vary ONE at a
    # time — a combined 2x*3x scaling would not distinguish which parameter is
    # used).
    base, _ = air_ice_stress(_arr(0.0), _arr(0.0), _arr(5.0), _arr(0.0),
                             _RHO_AIR, _C_AI)
    rho2, _ = air_ice_stress(_arr(0.0), _arr(0.0), _arr(5.0), _arr(0.0),
                             2.0 * _RHO_AIR, _C_AI)
    cd3, _ = air_ice_stress(_arr(0.0), _arr(0.0), _arr(5.0), _arr(0.0),
                            _RHO_AIR, 3.0 * _C_AI)
    np.testing.assert_allclose(_f(rho2) / _f(base), 2.0, rtol=1e-12)  # linear in rho_air
    np.testing.assert_allclose(_f(cd3) / _f(base), 3.0, rtol=1e-12)   # linear in C_ai


def test_ocean_stress_coefficient_plumbing():
    # Independent linear dependence on EACH of rho_ocean and C_oi (a hard-coded
    # default for either — the codex-flagged gap — would break its own ratio).
    base, _ = ocean_ice_stress(_arr(0.0), _arr(0.0), _arr(5.0), _arr(0.0),
                               _RHO_OCEAN, _C_OI)
    rho4, _ = ocean_ice_stress(_arr(0.0), _arr(0.0), _arr(5.0), _arr(0.0),
                               4.0 * _RHO_OCEAN, _C_OI)
    cd25, _ = ocean_ice_stress(_arr(0.0), _arr(0.0), _arr(5.0), _arr(0.0),
                               _RHO_OCEAN, 2.5 * _C_OI)
    np.testing.assert_allclose(_f(rho4) / _f(base), 4.0, rtol=1e-12)   # linear in rho_ocean
    np.testing.assert_allclose(_f(cd25) / _f(base), 2.5, rtol=1e-12)   # linear in C_oi


# ---------------------------------------------------------------------------
# 2. Free drift — Nansen-Zubov alpha.
# ---------------------------------------------------------------------------
def _alpha(rho_air=_RHO_AIR, c_ai=_C_AI, rho_oc=_RHO_OCEAN, c_oi=_C_OI):
    return np.sqrt(rho_air * c_ai / (rho_oc * c_oi))


def test_free_drift_alpha_and_interpolation():
    u_o, v_o, u_w, v_w = 0.2, 0.05, 8.0, -3.0
    ui, vi = free_drift_velocity(_arr(u_o), _arr(v_o), _arr(u_w), _arr(v_w),
                                 _C_OI, _C_AI, _RHO_AIR, _RHO_OCEAN)
    a = _alpha()
    np.testing.assert_allclose(_f(ui), u_o + a * (u_w - u_o), rtol=1e-12)
    np.testing.assert_allclose(_f(vi), v_o + a * (v_w - v_o), rtol=1e-12)


def test_free_drift_nansen_two_percent_rule():
    # With the CICE-default drags the ice drifts at ~1.68 % of the wind (relative
    # to the ocean) — the classic Nansen number.
    a = _alpha()
    assert 0.015 < a < 0.020
    np.testing.assert_allclose(a, 0.016807, rtol=1e-4)
    ui, _ = free_drift_velocity(_arr(0.0), _arr(0.0), _arr(10.0), _arr(0.0),
                                _C_OI, _C_AI, _RHO_AIR, _RHO_OCEAN)
    np.testing.assert_allclose(_f(ui) / 10.0, a, rtol=1e-12)   # ice speed / wind == alpha


def test_free_drift_no_relative_wind_gives_ocean_velocity():
    # Wind == ocean => no drift relative to the ocean.
    ui, vi = free_drift_velocity(_arr(0.3), _arr(-0.1), _arr(0.3), _arr(-0.1),
                                 _C_OI, _C_AI, _RHO_AIR, _RHO_OCEAN)
    np.testing.assert_allclose(_f(ui), 0.3, rtol=1e-12)
    np.testing.assert_allclose(_f(vi), -0.1, rtol=1e-12)


def test_free_drift_drag_plumbing():
    # Non-default DRAGS change alpha per the formula (a hard-coded default fails).
    c_oi, c_ai = 2.0e-3, 4.0e-3
    ui, _ = free_drift_velocity(_arr(0.0), _arr(0.0), _arr(10.0), _arr(0.0),
                                c_oi, c_ai, _RHO_AIR, _RHO_OCEAN)
    np.testing.assert_allclose(_f(ui) / 10.0, _alpha(c_ai=c_ai, c_oi=c_oi), rtol=1e-12)


def test_free_drift_density_plumbing():
    # Non-default DENSITIES change alpha per the formula too (codex-flagged gap:
    # an impl ignoring rho_air / rho_ocean would pass every other test).
    rho_air, rho_oc = 1.4, 1030.0
    ui, _ = free_drift_velocity(_arr(0.0), _arr(0.0), _arr(10.0), _arr(0.0),
                                _C_OI, _C_AI, rho_air, rho_oc)
    np.testing.assert_allclose(_f(ui) / 10.0,
                               _alpha(rho_air=rho_air, rho_oc=rho_oc), rtol=1e-12)


def test_free_drift_nansen_balance_with_quadratic_drag():
    # Consistency: at u_ice = alpha*W (ocean at rest) the actual air + ocean
    # stresses have the EXACT analytic residual of the wind>>ice approximation.
    # With tau_air ~ (1-a)^2 |W|W and tau_ocean = -rho_oc*C_oi*a^2 |W|W =
    # -rho_air*C_ai |W|W (since rho_oc*C_oi*a^2 == rho_air*C_ai), the normalized
    # residual is exactly (2a - a^2)/(1-a)^2 = 0.03448.  Uses the REAL stress fns,
    # so it cross-checks that alpha is the Nansen number consistent with the drags.
    u_w, v_w = 12.0, -5.0
    ui, vi = free_drift_velocity(_arr(0.0), _arr(0.0), _arr(u_w), _arr(v_w),
                                 _C_OI, _C_AI, _RHO_AIR, _RHO_OCEAN)
    tax, tay = air_ice_stress(ui, vi, _arr(u_w), _arr(v_w), _RHO_AIR, _C_AI)
    tox, toy = ocean_ice_stress(ui, vi, _arr(0.0), _arr(0.0), _RHO_OCEAN, _C_OI)
    resid = np.hypot(_f(tax) + _f(tox), _f(tay) + _f(toy))
    scale = np.hypot(_f(tax), _f(tay))
    a = _alpha()
    expected = (2.0 * a - a ** 2) / (1.0 - a) ** 2         # exact analytic residual
    # rtol 1e-6 (not round-off): the small ocean relative velocity (alpha*|W| ~ 0.2)
    # makes the 1e-10 floor a ~1e-9 relative perturbation on the ocean stress, which
    # the ~3.3 % near-cancellation in `resid` amplifies to ~3e-8.  Still a tight
    # 6-sig-fig pin of the analytic 0.0344810 — a wrong alpha gives a different ratio.
    np.testing.assert_allclose(resid / scale, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# 3. Signature defaults == config / constants (proven via omitted-arg calls).
# ---------------------------------------------------------------------------
def test_signature_defaults_match_config():
    # (A) Pin the actual bound default VALUES (Python evaluates defaults at import,
    # so inspect returns the numeric default).  Chained with
    # test_dynamics_config_defaults (config == CICE literals) this proves every
    # default coefficient equals the CICE value — a WRONG default (e.g. C_ai=1.5e-3)
    # fails here.  It does NOT distinguish `=_DYN_DEFAULTS.drag_atm` from an equal
    # hard-coded literal at runtime (both are the correct value); that is an
    # AST-level concern, not a faithfulness one.
    sa = inspect.signature(air_ice_stress).parameters
    assert sa["rho_air"].default == _RHO_AIR and sa["C_ai"].default == _C_AI
    so = inspect.signature(ocean_ice_stress).parameters
    assert so["rho_ocean"].default == _RHO_OCEAN and so["C_oi"].default == _C_OI
    sf = inspect.signature(free_drift_velocity).parameters
    assert sf["drag_ocean"].default == _C_OI and sf["drag_atm"].default == _C_AI
    assert sf["rho_air"].default == _RHO_AIR and sf["rho_ocean"].default == _RHO_OCEAN

    # (B) And the defaults are actually USED: omitting them reproduces the call
    # with the config/constant values passed explicitly.
    a1, a2 = air_ice_stress(_arr(0.0), _arr(0.0), _arr(5.0), _arr(-2.0))
    b1, b2 = air_ice_stress(_arr(0.0), _arr(0.0), _arr(5.0), _arr(-2.0),
                            _RHO_AIR, _C_AI)
    np.testing.assert_allclose([_f(a1), _f(a2)], [_f(b1), _f(b2)], rtol=1e-12)

    o1, o2 = ocean_ice_stress(_arr(0.0), _arr(0.0), _arr(0.3), _arr(0.1))
    p1, p2 = ocean_ice_stress(_arr(0.0), _arr(0.0), _arr(0.3), _arr(0.1),
                              _RHO_OCEAN, _C_OI)
    np.testing.assert_allclose([_f(o1), _f(o2)], [_f(p1), _f(p2)], rtol=1e-12)

    f1, f2 = free_drift_velocity(_arr(0.2), _arr(-0.1), _arr(8.0), _arr(3.0))
    g1, g2 = free_drift_velocity(_arr(0.2), _arr(-0.1), _arr(8.0), _arr(3.0),
                                 _C_OI, _C_AI, _RHO_AIR, _RHO_OCEAN)
    np.testing.assert_allclose([_f(f1), _f(f2)], [_f(g1), _f(g2)], rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. Differentiability at the speed floor.
# ---------------------------------------------------------------------------
def test_drag_gradient_finite_at_zero_relative_velocity():
    # |U_rel|*U_rel is mathematically C^1 at U_rel = 0 (its derivative there is 0),
    # but JAX's sqrt-then-multiply path hits 0/0 in d(sqrt) and returns NaN; the
    # added 1e-10 keeps that AD path finite when ice moves exactly with the flow.
    # Covers BOTH stress fns (each has its own floor).
    for stress, rho, cd in ((air_ice_stress, _RHO_AIR, _C_AI),
                            (ocean_ice_stress, _RHO_OCEAN, _C_OI)):
        def loss(u_i, stress=stress, rho=rho, cd=cd):
            tx, ty = stress(u_i, _arr(0.3), _arr(0.3), _arr(0.3), rho, cd)
            return jnp.sum(tx ** 2 + ty ** 2)
        g = jax.grad(loss)(_arr(0.3))     # U_rel == 0 exactly
        assert jnp.all(jnp.isfinite(g))


def test_free_drift_gradient_finite():
    def loss(u_w):
        ui, vi = free_drift_velocity(_arr(0.0), _arr(0.0), u_w, _arr(0.0),
                                     _C_OI, _C_AI, _RHO_AIR, _RHO_OCEAN)
        return jnp.sum(ui ** 2 + vi ** 2)
    g = jax.grad(loss)(_arr(8.0))
    assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# 5. Config defaults (CICE drag coefficients).
# ---------------------------------------------------------------------------
def test_dynamics_config_defaults():
    assert _CFG.drag_atm == 1.3e-3        # CICE air-ice drag C_ai
    assert _CFG.drag_ocean == 5.5e-3      # CICE ocean-ice drag C_oi
    assert constants.rho_air == 1.225
    assert constants.rho_ocean == 1025.0
