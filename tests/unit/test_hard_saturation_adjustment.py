"""Tests for the opt-in hard (bracketed-bisection) saturation adjustment.

Covers the ``hard_adjust`` path of
``legoesm.atmosphere.physics.microphysics._warm_rain.saturation_adjustment``
and its config plumbing.  The adjustment has two coupled parts:

* an iterated (bracketed-bisection + implicit-gradient) solve that finds the
  vapour ``q_eq`` ON the liquid saturation curve an enthalpy-conserving moist
  adjustment would reach (robust for any T, p, q_v incl. the q_sat cap and low
  pressure);
* a per-step latent-heating RATE LIMIT (``hard_max_heating_K``) so a large
  super-saturation pool drains onto that curve over MANY steps rather than
  dumping its full latent heat (~165 K for 67 g/kg) in one step -- the
  detonation the naive post-step cap caused in the field.

Invariants verified: (a) conservation of ``c_pd*T + L_v*q_v`` to machine
precision; (b) the UNCAPPED solve lands on the saturation curve for the
Tibet / low-pressure / extreme-q_v batteries; (c) with the default cap the
per-step heating is bounded and a 67 g/kg pool drains to saturation over many
steps, monotonically, without overshoot; (d) mild super-saturation (full drain
below the cap) still lands on the curve in one call; (e) flag-off bit-identity;
(f) AD is finite and (uncapped, mild) matches finite differences; plus
jit/vmap, the sub-saturated/dry sign check, and config/scheme threading.

Run with:

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/unit/test_hard_saturation_adjustment.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    saturation_adjustment,
)

_L_V = constants.L_v
_C_PD = constants.c_pd

# Dossier checkerboard (2026-07-23): Tibet plateau p_s = 654 hPa, cold/moderate
# cells hold the 66-67 g/kg vapour pool; 312.8/400.5 K are the HOT cells that
# are sub-saturated relative to the (capped) q_sat.
_P_TIBET = 65400.0
_Q_POOL = 0.066            # 66 g/kg
_DT = 100.0
_DEFAULT_MAX_HEATING_K = 5.0     # config default hard_sat_max_heating_K
# A cap large enough to disable the rate limit (isolates the on-curve solver).
_NO_CAP = 1.0e4
# Super-saturated checkerboard cells (RH >> threshold -> full hard adjustment).
_T_SUPERSAT = jnp.array(
    [115.9, 175.3, 200.2, 213.6, 217.6, 231.4, 237.3, 250.0, 270.0, 290.0]
)


def _apply(cond, T, q_v, dt):
    """Apply the condensation rate the way every warm-rain caller does."""
    q_v_new = q_v - cond * dt
    T_new = T + _L_V * cond * dt / _C_PD
    return T_new, q_v_new


# --------------------------------------------------------------------------
# (a) Conservation of moist enthalpy c_pd*T + L_v*q_v (rate-limited or not)
# --------------------------------------------------------------------------
def test_hard_adjustment_conserves_moist_enthalpy():
    T = _T_SUPERSAT
    q_v = jnp.full_like(T, _Q_POOL)
    p = jnp.full_like(T, _P_TIBET)
    cond, _ = saturation_adjustment(T, q_v, p, _DT, hard_adjust=True)
    T_new, q_v_new = _apply(cond, T, q_v, _DT)
    h0 = _C_PD * T + _L_V * q_v
    h1 = _C_PD * T_new + _L_V * q_v_new
    # Exact by construction (the caller maps the rate to +L_v/c_pd and -1);
    # machine precision in fp64.  Holds whether or not the rate limit binds.
    np.testing.assert_allclose(np.asarray(h1), np.asarray(h0),
                               rtol=1e-12, atol=1e-6)


# --------------------------------------------------------------------------
# (b) The UNCAPPED solve lands ON the saturation curve (Bug-1 regressions)
# --------------------------------------------------------------------------
def test_uncapped_solver_lands_on_curve_tibet():
    T = _T_SUPERSAT
    q_v = jnp.full_like(T, _Q_POOL)
    p = jnp.full_like(T, _P_TIBET)
    cond, _ = saturation_adjustment(T, q_v, p, _DT, hard_adjust=True,
                                    hard_max_heating_K=_NO_CAP)
    T_new, q_v_new = _apply(cond, T, q_v, _DT)
    rh_new = q_v_new / saturation_mixing_ratio(T_new, p)
    np.testing.assert_allclose(np.asarray(rh_new), 1.0, rtol=3e-3)
    # substantial drain when uncapped (RH from huge to ~1)
    assert jnp.all(q_v_new < q_v)
    assert float(q_v_new[0]) < 0.1 * _Q_POOL     # 115.9 K -> ~4.4 g/kg


def test_uncapped_solver_lands_on_curve_low_pressure():
    """Low pressure (model-top levels, p 100-1000 Pa): the steep q_sat curve is
    where a Newton step over-condenses (codex round-1: n=8 -> RH ~ 0.35 at
    200 K/100 Pa).  The bracketed bisection lands on the curve."""
    Ts = jnp.array([160.0, 180.0, 200.0, 215.0, 235.0])
    for p_val in (100.0, 200.0, 500.0, 1000.0):
        q_v = jnp.full_like(Ts, _Q_POOL)
        p = jnp.full_like(Ts, p_val)
        supersat = q_v > 1.1 * saturation_mixing_ratio(Ts, p)
        cond, _ = saturation_adjustment(Ts, q_v, p, _DT, hard_adjust=True,
                                        hard_max_heating_K=_NO_CAP)
        T_new, q_v_new = _apply(cond, Ts, q_v, _DT)
        rh_new = (q_v_new / saturation_mixing_ratio(T_new, p))[supersat]
        assert jnp.all(jnp.isfinite(rh_new))
        np.testing.assert_allclose(np.asarray(rh_new), 1.0, atol=5e-3,
                                   err_msg=f"off-curve at p={p_val} Pa")


def test_uncapped_solver_lands_on_curve_extreme_q_v():
    """Extreme super-saturation (q_v up to 1 kg/kg) sits at the q_sat cap edge
    -- where a Newton step overshoots into a sub-saturated state (codex rounds
    1-2).  The bracketed bisection lands on the curve for the full range."""
    for q_v_val in (0.1, 0.3, 0.6, 1.0):
        for p_val in (100.0, 200.0, 1000.0, 20000.0):
            T = jnp.array([150.0, 200.0, 250.0])
            q_v = jnp.full_like(T, q_v_val)
            p = jnp.full_like(T, p_val)
            supersat = q_v > 1.1 * saturation_mixing_ratio(T, p)
            cond, _ = saturation_adjustment(T, q_v, p, _DT, hard_adjust=True,
                                            hard_max_heating_K=_NO_CAP)
            T_new, q_v_new = _apply(cond, T, q_v, _DT)
            rh_new = (q_v_new / saturation_mixing_ratio(T_new, p))[supersat]
            assert jnp.all(jnp.isfinite(rh_new))
            np.testing.assert_allclose(
                np.asarray(rh_new), 1.0, atol=5e-3,
                err_msg=f"off-curve at q_v={q_v_val}, p={p_val} Pa")


def test_uncapped_smooth_overshoots_hard_lands_on_curve_cold():
    """Strong (cold) super-saturation: the smooth sigmoid evaluates the
    psychrometric factor at the ORIGINAL cold T (dq_sat/dT ~ 0) and OVERSHOOTS
    past saturation into a bone-dry state; the (uncapped) hard solve lands ON
    the curve."""
    T = jnp.array([200.0])
    q_v = jnp.array([_Q_POOL])
    p = jnp.array([_P_TIBET])
    cs, _ = saturation_adjustment(T, q_v, p, _DT, hard_adjust=False)
    ch, _ = saturation_adjustment(T, q_v, p, _DT, hard_adjust=True,
                                  hard_max_heating_K=_NO_CAP)
    _, qv_s = _apply(cs, T, q_v, _DT)
    _, qv_h = _apply(ch, T, q_v, _DT)
    T_s, _ = _apply(cs, T, q_v, _DT)
    T_h, _ = _apply(ch, T, q_v, _DT)
    rh_s = float(qv_s[0] / saturation_mixing_ratio(T_s, p)[0])
    rh_h = float(qv_h[0] / saturation_mixing_ratio(T_h, p)[0])
    assert rh_h == pytest.approx(1.0, rel=3e-3)   # hard: ON the curve
    assert rh_s < 0.5                             # smooth: overshoots (too dry)


# --------------------------------------------------------------------------
# (c) Per-step RATE LIMIT + multi-step draining (the field-detonation fix)
# --------------------------------------------------------------------------
def test_per_step_latent_heating_is_rate_limited():
    """With the default cap, the per-call latent heating never exceeds
    hard_max_heating_K for strongly super-saturated cells (which would otherwise
    dump ~140 K in one step)."""
    T = jnp.array([150.0, 200.0, 237.0, 270.0, 290.0])
    q_v = jnp.full_like(T, _Q_POOL)
    p = jnp.full_like(T, _P_TIBET)
    for dt in (30.0, 100.0, 300.0, 1200.0):
        cond, _ = saturation_adjustment(T, q_v, p, dt, hard_adjust=True)
        dT = _L_V * cond * dt / _C_PD
        # per-step heating <= cap (the cap is a per-step Delta q_v / Delta T,
        # dt-independent), for every dt.
        assert jnp.all(dT <= _DEFAULT_MAX_HEATING_K + 1e-6), f"dt={dt}"
        # still draining (positive condensation).
        assert jnp.all(cond > 0.0)


@pytest.mark.parametrize("T0,q_v0,min_steps,min_dT_rise", [
    (230.0, 0.067, 12, 30.0),   # cold pool (dossier checkerboard): many steps
    (260.0, 0.10, 8, 30.0),     # moderate pool: many steps
    (320.0, 0.20, 1, 3.0),      # WARM pool (codex round-4 overshoot case):
                                # q_sat ~ 0.12 there, so only ~3 g/kg drains in a
                                # few steps -- the point is NO OVERSHOOT.
])
def test_large_pool_drains_to_saturation_over_many_steps(T0, q_v0, min_steps,
                                                         min_dT_rise):
    """The coordinator's required test: a large super-saturation column drains
    to saturation over MULTIPLE calls with per-call Delta T <= the bound,
    monotonically, and WITHOUT overshooting below saturation -- for cold AND
    warm pools (the warm pool is where the finite smooth blend used to overshoot
    to RH ~ 0.999 and re-evaporate; codex round-4)."""
    T = jnp.array([T0])
    p = jnp.array([_P_TIBET])
    q_v = jnp.array([q_v0])
    dt = 100.0
    q_prev = float(q_v[0])
    T_start = float(T[0])
    n_steps_to_converge = None
    for step in range(160):
        cond, _ = saturation_adjustment(T, q_v, p, dt, hard_adjust=True)
        dq = cond * dt
        dT = _L_V * cond * dt / _C_PD
        # (i) per-step heating bounded (no one-step detonation).
        assert float(dT[0]) <= _DEFAULT_MAX_HEATING_K + 1e-6
        # (ii) monotone drain: q_v only decreases (never re-evaporates).
        assert float(dq[0]) >= -1e-12, f"re-evaporation at step {step}"
        T, q_v = _apply(cond, T, q_v, dt)
        assert float(q_v[0]) <= q_prev + 1e-12
        q_prev = float(q_v[0])
        # (iii) approaches saturation strictly FROM ABOVE -- NEVER overshoots
        # below (tight tolerance: the on-curve cap forbids the smooth overshoot).
        rh = float((q_v / saturation_mixing_ratio(T, p))[0])
        assert rh >= 1.0 - 1e-4, f"overshoot below saturation at step {step}"
        if n_steps_to_converge is None and abs(rh - 1.0) < 1e-2:
            n_steps_to_converge = step
    # It reached saturation, in more than ONE step (rate-limited, not a
    # one-step detonation), taking >= the case-appropriate number of steps.
    assert n_steps_to_converge is not None
    assert n_steps_to_converge >= min_steps
    rh_final = float((q_v / saturation_mixing_ratio(T, p))[0])
    np.testing.assert_allclose(rh_final, 1.0, atol=1e-2)
    # The pool was substantially drained (T rose from the released latent heat).
    assert float(T[0]) - T_start >= min_dT_rise


# --------------------------------------------------------------------------
# (d) Mild super-saturation (full drain below the cap) lands on curve in 1 call
# --------------------------------------------------------------------------
def test_mild_supersaturation_lands_on_curve_in_one_call():
    """Where the full drain-to-curve Delta q_v is below the per-step cap, the
    rate limit is inactive and the cell lands ON the curve in a single call
    (psychrometric consistency for the common case)."""
    # Cool cells with small q_sat => small absolute excess (full drain < cap)
    # even at RH 1.6, which is far enough above the threshold that the smooth
    # -> hard activation is ~ 1 (so the blend lands ON the curve).
    T = jnp.array([248.0, 252.0, 256.0, 260.0])
    p = jnp.full_like(T, _P_TIBET)
    q_sat0 = saturation_mixing_ratio(T, p)
    q_v = 1.6 * q_sat0
    cond, _ = saturation_adjustment(T, q_v, p, _DT, hard_adjust=True)
    dT = _L_V * cond * _DT / _C_PD
    assert jnp.all(dT < _DEFAULT_MAX_HEATING_K)   # cap inactive (mild)
    T_new, q_v_new = _apply(cond, T, q_v, _DT)
    rh_new = q_v_new / saturation_mixing_ratio(T_new, p)
    np.testing.assert_allclose(np.asarray(rh_new), 1.0, rtol=3e-3)


# --------------------------------------------------------------------------
# (e) Flag-off bit-identity
# --------------------------------------------------------------------------
def test_flag_off_bit_identical():
    T = jnp.array([[285.0, 250.0], [300.0, 200.0]])
    q_v = jnp.array([[0.02, 0.01], [0.03, 0.05]])
    p = jnp.full_like(T, 8.0e4)
    q_c = jnp.full_like(T, 5.0e-4)
    ref_c, ref_q = saturation_adjustment(T, q_v, p, 300.0, q_c=q_c)
    off_c, off_q = saturation_adjustment(T, q_v, p, 300.0, q_c=q_c,
                                         hard_adjust=False)
    assert jnp.array_equal(ref_c, off_c)
    assert jnp.array_equal(ref_q, off_q)
    # off is a no-op regardless of the threshold / heating-cap values.
    off2_c, _ = saturation_adjustment(T, q_v, p, 300.0, q_c=q_c,
                                      hard_adjust=False, hard_threshold=1.5,
                                      hard_max_heating_K=2.0)
    assert jnp.array_equal(ref_c, off2_c)


# --------------------------------------------------------------------------
# (f) Differentiability
# --------------------------------------------------------------------------
def test_hard_adjustment_grad_finite_and_nonzero():
    # Low-pressure column (bisection), extreme q_v (cap edge), a dry cell, and a
    # rate-limited cell -> every code path exercised.
    T = jnp.array([200.0, 150.0, 235.0, 320.0, 230.0])
    q_v = jnp.array([_Q_POOL, 1.0, _Q_POOL, 0.0, 0.067])
    p = jnp.array([_P_TIBET, 200.0, 500.0, _P_TIBET, _P_TIBET])

    def loss(T, q_v, p):
        cond, _ = saturation_adjustment(T, q_v, p, _DT, hard_adjust=True)
        return jnp.sum(cond ** 2)

    gT, gq, gp = jax.grad(loss, argnums=(0, 1, 2))(T, q_v, p)
    for g in (gT, gq, gp):
        assert jnp.all(jnp.isfinite(g)), "non-finite gradient through hard adj"
    assert float(jnp.sum(jnp.abs(gq))) > 0.0


def test_uncapped_grad_matches_finite_difference():
    """Centred FD vs autodiff of the (uncapped, mild) condensation w.r.t. q_v,
    T and p -- the implicit-function-theorem gradient of the bisection root."""
    T0 = 288.0
    p0 = 9.0e4
    q_sat0 = saturation_mixing_ratio(jnp.array([T0]), jnp.array([p0]))
    q_v0 = float(1.4 * q_sat0[0])       # RH 140 %

    def cond_of(T, q_v, p):
        c, _ = saturation_adjustment(jnp.array([T]), jnp.array([q_v]),
                                     jnp.array([p]), _DT, hard_adjust=True,
                                     hard_max_heating_K=_NO_CAP)
        return jnp.sum(c)

    for name, x0, fn in [
        ("q_v", q_v0, lambda a: cond_of(T0, a, p0)),
        ("T", T0, lambda a: cond_of(a, q_v0, p0)),
        ("p", p0, lambda a: cond_of(T0, q_v0, a)),
    ]:
        g = float(jax.grad(fn)(x0))
        h = 1.0e-6 * max(abs(x0), 1.0)
        fd = float((fn(x0 + h) - fn(x0 - h)) / (2.0 * h))
        np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-9,
                                   err_msg=f"grad/FD mismatch d/d{name}")


# --------------------------------------------------------------------------
# jit / vmap consistency
# --------------------------------------------------------------------------
def test_hard_adjustment_jit_and_vmap_match_eager():
    T = _T_SUPERSAT.reshape(1, -1)
    q_v = jnp.full_like(T, _Q_POOL)
    p = jnp.full_like(T, _P_TIBET)

    def fn(T, q_v, p):
        return saturation_adjustment(T, q_v, p, _DT, hard_adjust=True)[0]

    eager = fn(T, q_v, p)
    jitted = jax.jit(fn)(T, q_v, p)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager),
                               rtol=1e-6, atol=1e-12)
    Tb = jnp.stack([T, T + 5.0])
    qb = jnp.stack([q_v, q_v])
    pb = jnp.stack([p, p])
    vmapped = jax.vmap(fn)(Tb, qb, pb)
    np.testing.assert_allclose(np.asarray(vmapped[0]), np.asarray(eager),
                               rtol=1e-6, atol=1e-12)


# --------------------------------------------------------------------------
# q_sat cap region + sign correctness
# --------------------------------------------------------------------------
def test_cap_region_drains_vapour_above_capped_qsat():
    """Where q_v exceeds hard_threshold * (capped) q_sat, the guard drains
    (rate-limited) toward the cap -- the region a naive post-step cap missed.
    Uncapped, it reaches the cap in one call."""
    T = jnp.array([400.5, 380.0])
    q_v = jnp.array([1.3, 1.5])          # kg/kg, blowup magnitudes
    p = jnp.full_like(T, _P_TIBET)
    q_sat = saturation_mixing_ratio(T, p)          # smooth-capped ~ 1 kg/kg
    # rate-limited: positive drain, heating bounded.
    cond, _ = saturation_adjustment(T, q_v, p, _DT, hard_adjust=True)
    assert jnp.all(cond > 0.0)
    assert jnp.all(_L_V * cond * _DT / _C_PD <= _DEFAULT_MAX_HEATING_K + 1e-6)
    # uncapped: reaches ~ the cap in one call.
    cond_u, _ = saturation_adjustment(T, q_v, p, _DT, hard_adjust=True,
                                      hard_max_heating_K=_NO_CAP)
    _, q_v_new = _apply(cond_u, T, q_v, _DT)
    assert jnp.all(q_v_new < q_v - 0.1)
    assert jnp.all(q_v_new <= 1.05 * q_sat)


def test_hot_subsaturated_cell_not_condensed():
    """At 400/313 K the 66 g/kg pool is BELOW the (capped) q_sat: sub-saturated.
    The guard must NOT manufacture condensation there (no sign inversion)."""
    T = jnp.array([400.5, 312.8])
    q_v = jnp.full_like(T, _Q_POOL)
    p = jnp.full_like(T, _P_TIBET)
    cond, _ = saturation_adjustment(T, q_v, p, _DT, hard_adjust=True)
    assert jnp.all(cond <= 1.0e-5)


def test_no_nan_on_extreme_and_dry_inputs():
    T = jnp.array([[115.9, 237.3, 312.8, 400.5, 288.0]])
    q_v = jnp.array([[0.066, 0.066, 0.066, 0.066, 0.0]])
    p = jnp.full_like(T, _P_TIBET)
    q_c = jnp.zeros_like(T)
    cond, q_sat = saturation_adjustment(T, q_v, p, _DT, q_c=q_c,
                                        hard_adjust=True)
    assert jnp.all(jnp.isfinite(cond))
    assert jnp.all(jnp.isfinite(q_sat))


def test_mass_positivity_cold_dry_column():
    """``saturation_mixing_ratio`` returns a tiny NEGATIVE value (~ -2e-11) at
    very cold T from the smooth upper cap (codex round-3 finding).  The
    adjustment must never remove more vapour than the column holds -- q_v stays
    >= 0 -- for dry / marginal / cold cells across a range of dt."""
    T = jnp.array([150.0, 150.0, 160.0, 175.0, 200.0])
    q_v = jnp.array([0.0, 1.0e-11, 1.0e-8, 1.0e-5, 0.0])
    p = jnp.full_like(T, _P_TIBET)
    for dt in (30.0, 100.0, 1200.0):
        cond, _ = saturation_adjustment(T, q_v, p, dt, hard_adjust=True)
        q_v_new = q_v - cond * dt
        assert jnp.all(jnp.isfinite(cond))
        # mass-positivity: never drive vapour negative.
        assert jnp.all(q_v_new >= -1e-20), f"q_v<0 at dt={dt}: {q_v_new}"


# --------------------------------------------------------------------------
# Config plumbing: experiment-flag threading + scheme-level behaviour
# --------------------------------------------------------------------------
def test_hard_saturation_drain_is_pure_gated_drain():
    """The driver post-step entry point ``hard_saturation_drain``: a PURE
    activation-gated drain (>= 0; no smooth condensation/evaporation baseline),
    rate-limited, conserving -- reusing the same reviewed core as the in-scheme
    path (smooth baseline = 0)."""
    from legoesm.atmosphere.physics.microphysics._warm_rain import (
        hard_saturation_drain,
    )
    # supersat / subsat / hot-subsat / supersat / MILD (RH ~ 1.05, below the
    # 1.1 trigger -> the HARD gate must NOT leak there).
    T = jnp.array([200.0, 290.0, 312.8, 250.0, 260.0])
    p = jnp.full_like(T, _P_TIBET)
    q_mild = 1.05 * saturation_mixing_ratio(jnp.array([260.0]), p[:1])[0]
    q_v = jnp.array([_Q_POOL, 0.001, _Q_POOL, _Q_POOL, float(q_mild)])
    dq = hard_saturation_drain(T, q_v, p, _DT) * _DT
    # PURE drain: never negative (no evaporation of sub-saturated cells).
    assert jnp.all(dq >= 0.0)
    q_sat = saturation_mixing_ratio(T, p)
    # HARD gate: everything at or below 1.1*q_sat is UNTOUCHED (no leak),
    # including the mild RH ~ 1.05 cell.
    at_or_below = q_v <= 1.1 * q_sat
    assert jnp.all(dq[at_or_below] <= 1e-12)
    assert float(dq[4]) == 0.0                    # RH ~ 1.05 -> exactly 0 (no leak)
    supersat = q_v > 1.1 * q_sat
    assert jnp.all(dq[supersat] > 0.0)            # super-saturated -> drains
    # rate-limited to the per-step heating cap (5 K -> ~2 g/kg).
    dqmax = _DEFAULT_MAX_HEATING_K * _C_PD / _L_V
    assert jnp.all(dq <= dqmax + 1e-12)
    # applying it conserves c_pd*T + L_v*q_v.
    T_new = T + _L_V * dq / _C_PD
    q_v_new = q_v - dq
    np.testing.assert_allclose(
        np.asarray(_C_PD * T_new + _L_V * q_v_new),
        np.asarray(_C_PD * T + _L_V * q_v), rtol=1e-12, atol=1e-6)


def test_mpas_poststep_hook_conserves_and_rate_limits():
    """Driver post-step hook ``_mpas_hard_saturation_poststep`` on a fabricated
    supersaturated column: conservation of c_pd*T + L_v*q_v AND total water,
    per-step heating <= cap, pure drain, donor-consistent q_v->q_c."""
    from legoesm.driver.model_driver import _mpas_hard_saturation_poststep
    ncol, nlev = 2, 5
    T = jnp.array([[200.0, 237.0, 260.0, 290.0, 312.8],   # cold pool + hot cell
                   [280.0, 285.0, 290.0, 295.0, 300.0]])  # sub-saturated column
    q_v = jnp.array([[0.067, 0.067, 0.067, 0.02, 0.067],
                     [0.001, 0.002, 0.003, 0.004, 0.005]])
    q_c = jnp.zeros((ncol, nlev))
    p_s = jnp.array([65400.0, 100000.0])
    sigma_full = jnp.linspace(0.95, 0.2, nlev)
    dt = 100.0
    T_new, q_v_new, q_c_new, dq = _mpas_hard_saturation_poststep(
        T, q_v, q_c, p_s, sigma_full, dt, 1.1, 5.0)
    assert jnp.all(dq >= 0.0)                           # pure drain
    dT = _L_V * dq / _C_PD
    assert jnp.all(dT <= 5.0 + 1e-6)                    # per-step heating <= cap
    # conservation of c_pd*T + L_v*q_v
    np.testing.assert_allclose(
        np.asarray(_C_PD * T_new + _L_V * q_v_new),
        np.asarray(_C_PD * T + _L_V * q_v), rtol=1e-12, atol=1e-6)
    # total water q_v + q_c conserved; q_c gains exactly what q_v loses
    np.testing.assert_allclose(np.asarray(q_v_new + q_c_new),
                               np.asarray(q_v + q_c), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.asarray(q_c_new - q_c),
                               np.asarray(q_v - q_v_new), rtol=1e-12, atol=1e-15)
    # the cold super-saturated pool actually drained; the sub-saturated column
    # (row 1) is untouched.
    assert float(jnp.sum(dq[0])) > 0.0
    assert float(jnp.sum(dq[1])) == pytest.approx(0.0, abs=1e-12)
    # None-q_c path is a conserving NO-OP: with no reservoir for the condensate,
    # the drain must not remove vapour (that would LOSE total water).
    T2, qv2, qc2, dq2 = _mpas_hard_saturation_poststep(
        T, q_v, None, p_s, sigma_full, dt, 1.1, 5.0)
    assert qc2 is None
    np.testing.assert_allclose(np.asarray(qv2), np.asarray(q_v), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(T2), np.asarray(T), rtol=1e-12)
    assert float(jnp.sum(dq2)) == 0.0


def test_experiment_flag_threads_and_raises():
    from legoesm.atmosphere.physics.microphysics.config import (
        apply_microphysics_experiment_flags,
        MorrisonConfig,
        SundqvistConfig,
    )
    on = apply_microphysics_experiment_flags(
        MorrisonConfig(), "morrison", hard_saturation_adjustment=True)
    assert on.hard_saturation_adjustment is True
    off = apply_microphysics_experiment_flags(
        MorrisonConfig(), "morrison", hard_saturation_adjustment=False)
    assert off.hard_saturation_adjustment is False
    assert off == MorrisonConfig()
    with pytest.raises(ValueError, match="hard_saturation_adjustment"):
        apply_microphysics_experiment_flags(
            SundqvistConfig(), "sundqvist", hard_saturation_adjustment=True)


def test_all_warm_rain_configs_carry_the_fields():
    from legoesm.atmosphere.physics.microphysics.config import (
        KesslerConfig, SeifertBehengConfig, MorrisonConfig, ThompsonConfig,
        P3Config,
    )
    for cfg in (KesslerConfig(), SeifertBehengConfig(), MorrisonConfig(),
                ThompsonConfig(), P3Config()):
        assert cfg.hard_saturation_adjustment is False
        assert cfg.hard_sat_adjust_threshold == pytest.approx(1.1)
        assert cfg.hard_sat_max_heating_K == pytest.approx(5.0)


def test_kessler_scheme_threads_flag_and_bounds_heating():
    from legoesm.atmosphere.physics.microphysics.kessler import (
        kessler_microphysics,
    )
    from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState

    ncol, nlev = 1, 4
    T = jnp.array([[200.0, 230.0, 260.0, 285.0]])
    p = jnp.full((ncol, nlev), _P_TIBET)
    rho = p / (constants.R_d * T)
    q_v = jnp.full((ncol, nlev), 0.03)         # super-saturated at cold levels
    dz = jnp.full((ncol, nlev), 500.0)
    p_half = jnp.concatenate([p, 1.01 * p[:, -1:]], axis=1)
    z = jnp.zeros((ncol, nlev))
    hydro = HydrometeorState(
        q_c=z, q_r=z, q_i=z, q_s=z, q_g=z,
        N_c=jnp.full((ncol, nlev), 1.0e8), N_r=z, N_i=z,
    )
    dt = 100.0
    out_off = kessler_microphysics(T, q_v, hydro, p, p_half, rho, dz, dt,
                                   KesslerConfig())
    out_off2 = kessler_microphysics(
        T, q_v, hydro, p, p_half, rho, dz, dt,
        KesslerConfig(hard_saturation_adjustment=False))
    out_on = kessler_microphysics(
        T, q_v, hydro, p, p_half, rho, dz, dt,
        KesslerConfig(hard_saturation_adjustment=True))
    # Flag off == default config, byte-for-byte.
    assert jnp.array_equal(out_off.dq_v_dt, out_off2.dq_v_dt)
    # The flag changes the result (it is wired in).
    assert not jnp.array_equal(out_off.dq_v_dt, out_on.dq_v_dt)
    # The hard path's per-step latent heating is bounded (no detonation): the
    # saturation-adjustment contribution alone heats <= the cap.  (Kessler's
    # dT_dt also includes rain evaporation cooling, so bound the condensation
    # part via dq_v.)  Here q_c starts at 0 so condensation = -dq_v_sat > 0.
    assert jnp.all(out_on.dq_v_dt * dt >= -(_DEFAULT_MAX_HEATING_K
                                            * _C_PD / _L_V) - 1e-9)
    # The hard path removes strictly LESS vapour per step than the smooth path's
    # cold-cell over-condensation (rate-limited) -> leaves >= as much vapour.
    q_v_off = q_v + out_off.dq_v_dt * dt
    q_v_on = q_v + out_on.dq_v_dt * dt
    assert jnp.all(q_v_on >= q_v_off - 1e-6)
    # Both paths conserve c_pd*T + L_v*q_v (warm, no ice).
    for out in (out_off, out_on):
        h0 = _C_PD * T + _L_V * q_v
        h1 = _C_PD * (T + out.dT_dt * dt) + _L_V * (q_v + out.dq_v_dt * dt)
        np.testing.assert_allclose(np.asarray(h1), np.asarray(h0),
                                   rtol=1e-9, atol=1e-3)
