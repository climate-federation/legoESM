"""Scheme-level oracle-faithfulness tests for the diagnostic cloud fraction.

Pins the Xu-Randall (1996) and Sundqvist-Berge-Kristjansson (1989) cloud-fraction
forms and locks the AD-safety departures as canaries.

FAITHFUL (paper forms + published constants):
  * Xu-Randall (1996) Eq. (6) cf = RH^p·[1 − exp(−α·q_c/((1−RH)·q_sat)^γ)] with the
    PAPER constants α=100, p=0.25, γ=0.49 — the guarded production form pinned
    EXACTLY against an independent NumPy reimplementation carrying the SAME floors
    (it reduces to bare Eq. (6) where the guards are inactive);
  * Sundqvist-Berge-Kristjansson (1989) √-form cf = 1 − √((1−RH)/(1−RH_crit)) for
    RH≥RH_crit (else 0), 1 at RH≥1 — pinned exactly (NOT a linear RH ramp).

DEPARTURE (locked + labeled):
  * AD guards (Xu-Randall): the (1−RH)·q_sat denominator floor (1e-10) and the
    RH∈[1e-6,1] clip make the fractional-power (γ<1, p<1) reverse-mode DERIVATIVES
    finite; for physical inputs (q_c ≥ 0, q_sat > 0) the forward value equals bare
    Eq. (6) wherever both guards are inactive (a non-vacuity check shows the
    un-guarded derivative is non-finite);
  * boundary enforcement: both cloud fractions clipped to [0, 1] (this IS
    Sundqvist's RH<RH_crit→0 branch; redundant-by-construction for Xu-Randall);
  * model choices: rh_crit=0.77 default (canary) and the linear T ice-fraction
    ramp (endpoint/midpoint canary).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    _ice_fraction,
    sundqvist_cloud_fraction,
    xu_randall_cloud_fraction,
)
from legoesm.atmosphere.physics.clouds.config import CloudConfig


@pytest.fixture(autouse=True)
def _enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _xu_randall_numpy(RH, q_c, q_s, cfg):
    """Independent reimplementation of the documented Xu-Randall form + floors."""
    RH = np.asarray(RH, dtype=np.float64)
    q_c = np.asarray(q_c, dtype=np.float64)
    q_s = np.asarray(q_s, dtype=np.float64)
    denom = np.maximum((1.0 - RH) * q_s, 1.0e-10) ** cfg.gamma_xr
    expo = -cfg.alpha_xr * q_c / denom
    cf = np.clip(RH, 1.0e-6, 1.0) ** cfg.p_xr * (1.0 - np.exp(expo))
    return np.clip(cf, 0.0, 1.0)


# ===========================================================================
# FAITHFUL forms
# ===========================================================================
def test_faithful_xu_randall_paper_constants():
    """Canary: the Xu-Randall (1996) semiempirical constants are the paper values."""
    c = CloudConfig()
    assert c.alpha_xr == 100.0   # Xu & Randall (1996) alpha_0
    assert c.p_xr == 0.25        # Xu & Randall (1996) p
    assert c.gamma_xr == 0.49    # Xu & Randall (1996) gamma


def test_faithful_xu_randall_form_pinned_exactly():
    """xu_randall_cloud_fraction == an independent reimpl of the GUARDED form.

    The reimpl carries the same floors, so on the guard-inactive interior it is
    bare paper Eq. (6); the grid below deliberately also hits both guards.
    Grid spans sub-critical, resolved, saturated, and super-saturated RH plus a
    range of condensate/q_sat. The RH=0 point carries q_c>0 so the RH∈[1e-6,1]
    clip is OBSERVABLY exercised (RH^p at 1e-6 gives a nonzero factor times a
    nonzero 1−exp), and the RH>1 points drive (1−RH)·q_sat<0 so the 1e-10
    denominator floor is exercised too.
    """
    cfg = CloudConfig()
    RH = jnp.array([[0.0, 0.5, 0.78, 0.9, 0.99, 1.0, 1.05]])
    q_c = jnp.array([[1e-4, 1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 1e-3]])
    q_s = jnp.array([[1e-2, 1e-2, 8e-3, 6e-3, 4e-3, 3e-3, 3e-3]])
    got = np.asarray(xu_randall_cloud_fraction(RH, q_c, q_s, cfg))
    exp = _xu_randall_numpy(RH, q_c, q_s, cfg)
    assert np.allclose(got, exp, rtol=1e-12, atol=1e-14)
    # the RH=0 point is nonzero (RH clip active: 1e-6^p * (1-exp) > 0)
    assert float(got[0, 0]) > 0.0
    # a resolved mid-RH point genuinely lands strictly inside (0, 1) (not a clip)
    assert 0.0 < float(got[0, 3]) < 1.0


def test_faithful_xu_randall_monotone_and_condensate_limit():
    """cf rises with RH and with condensate; cf → 0 as q_c → 0 (the exp factor)."""
    cfg = CloudConfig()
    q_s = jnp.full((1, 5), 6.0e-3)
    q_c = jnp.full((1, 5), 5.0e-4)
    RH = jnp.array([[0.80, 0.85, 0.90, 0.95, 0.99]])
    cf = np.asarray(xu_randall_cloud_fraction(RH, q_c, q_s, cfg))[0]
    assert np.all(np.diff(cf) >= -1e-12)              # non-decreasing in RH
    # non-decreasing in condensate at fixed RH
    RH1 = jnp.full((1, 5), 0.9)
    qc_grid = jnp.array([[0.0, 1e-4, 3e-4, 6e-4, 1e-3]])
    cf_q = np.asarray(xu_randall_cloud_fraction(RH1, qc_grid, q_s, cfg))[0]
    assert np.all(np.diff(cf_q) >= -1e-12)
    assert cf_q[0] == pytest.approx(0.0)             # q_c=0 -> 1-exp(0)=0 -> cf=0


def test_faithful_sundqvist_sqrt_form_pinned_exactly():
    """sundqvist_cloud_fraction == 1 − √((1−RH)/(1−RH_crit)) (NOT a linear ramp)."""
    cfg = CloudConfig()
    RH = jnp.array([[0.0, 0.5, 0.77, 0.85, 0.95, 1.0, 1.1]])
    got = np.asarray(sundqvist_cloud_fraction(RH, cfg))
    RHn = np.asarray(RH)
    arg = (1.0 - RHn) / max(1.0 - cfg.rh_crit, 1.0e-6)
    exp = np.where(arg > 0.0, 1.0 - np.sqrt(np.where(arg > 0.0, arg, 1.0)), 1.0)
    exp = np.clip(exp, 0.0, 1.0)
    assert np.allclose(got, exp, rtol=1e-12, atol=1e-14)
    # below rh_crit -> 0 ; at/above saturation -> 1
    assert float(got[0, 1]) == pytest.approx(0.0)    # RH=0.5 < rh_crit
    assert float(got[0, 5]) == pytest.approx(1.0)    # RH=1.0
    # the sqrt-form is NOT the linear ramp: at the RH-midpoint of [rh_crit, 1] the
    # sqrt-form is 1-sqrt(1/2)=0.293, BELOW the linear ramp's 0.5. The sqrt-form
    # stays below the linear ramp on the whole interior; its slope at rh_crit is
    # 0.5/(1-rh_crit) (HALF the linear ramp's 1/(1-rh_crit)) and diverges only as
    # RH -> 1-.
    RH_mid = 0.5 * (cfg.rh_crit + 1.0)
    cf_mid = float(np.asarray(sundqvist_cloud_fraction(jnp.array([[RH_mid]]), cfg))[0, 0])
    linear_mid = (RH_mid - cfg.rh_crit) / (1.0 - cfg.rh_crit)   # == 0.5
    assert cf_mid == pytest.approx(1.0 - np.sqrt(0.5))          # the sqrt-form value
    assert cf_mid < linear_mid - 0.1                            # strictly below linear


# ===========================================================================
# DEPARTURES (AD-safety floors / clips)
# ===========================================================================
def test_departure_xu_randall_gradients_finite_at_floors():
    """The RH clip + denominator floor keep d(cf)/d(RH), d(cf)/d(q_c) finite.

    Without the [1e-6,1] RH clip, RH^p (p<1) has an infinite derivative at RH=0.
    Without the 1e-10 denominator floor, the denominator ((1−RH)q_s)^γ → 0 as
    (1−RH)q_s → 0, so for fixed q_c>0 the quotient α·q_c/((1−RH)q_s)^γ → ∞; at RH=1
    the raw un-floored expression is undefined (0/0 if q_c=0, else ∞) — JAX then
    produces a non-finite AD result.
    Both would give inf/NaN reverse-mode cotangents on dry/saturated layers.
    """
    cfg = CloudConfig()

    def cf_of_RH(rh):
        return xu_randall_cloud_fraction(
            jnp.array([[rh]]), jnp.array([[1e-4]]), jnp.array([[1e-2]]), cfg)[0, 0]

    # d(cf)/d(q_c) evaluated at RH=1 exactly, where (1-RH)*q_s = 0 is floored to
    # 1e-10 (the denominator floor is ACTIVE here, not merely small).
    def cf_of_qc_at_sat(qc):
        return xu_randall_cloud_fraction(
            jnp.array([[1.0]]), jnp.array([[qc]]), jnp.array([[1e-2]]), cfg)[0, 0]

    assert np.isfinite(float(jax.grad(cf_of_RH)(0.0)))         # RH=0 dry layer (RH clip)
    assert np.isfinite(float(jax.grad(cf_of_RH)(1.0)))         # RH=1 saturation
    assert np.isfinite(float(jax.grad(cf_of_qc_at_sat)(1e-4)))  # denom floor active

    # Non-vacuity (RH clip): the SAME form WITHOUT the RH clip has a non-finite
    # RH=0 gradient — RH^p (p<1) has an infinite DERIVATIVE at RH=0 (the forward
    # value 0^p=0 is finite; only the derivative diverges).
    def cf_no_rh_clip(rh):
        denom = ((1.0 - rh) * 1e-2) ** cfg.gamma_xr
        return rh ** cfg.p_xr * (1.0 - jnp.exp(-cfg.alpha_xr * 1e-4 / denom))

    assert not np.isfinite(float(jax.grad(cf_no_rh_clip)(0.0)))

    # Non-vacuity (denominator floor): WITHOUT the 1e-10 floor, at RH=1 the base
    # (1-RH)*q_s = 0 makes ((1-RH)q_s)^gamma = 0, so alpha*q_c/0 -> the exponent
    # and its q_c-DERIVATIVE are non-finite.
    def cf_no_denom_floor(qc):
        denom = ((1.0 - 1.0) * 1e-2) ** cfg.gamma_xr           # 0 ** gamma == 0
        return jnp.clip(jnp.array(1.0), 1e-6, 1.0) ** cfg.p_xr * (
            1.0 - jnp.exp(-cfg.alpha_xr * qc / denom))

    assert not np.isfinite(float(jax.grad(cf_no_denom_floor)(1e-4)))


def test_departure_clipped_to_unit_interval():
    """Both cloud fractions stay in [0, 1] under extreme inputs (clip departure).

    Non-vacuity for SUNDQVIST: the un-clipped √-form genuinely goes below 0 for
    RH < RH_crit (e.g. RH=−0.2 → 1−√5.2 < 0), so its [0,1] clip is actually firing.
    For XU-RANDALL the [0,1] clip is REDUNDANT/defensive FOR PHYSICAL INPUTS
    (q_c ≥ 0, q_sat > 0): cf = clip(RH,1e-6,1)^p · (1−exp(≤0)) is then a product of
    two factors each in [0,1], so cf ∈ [0,1] BY CONSTRUCTION (the upper end reached
    when exp underflows to 0 at saturation) — the test documents this rather than
    claiming its clip fires. (A negative q_c would flip the exp-argument sign and
    could drive cf<0, where the clip WOULD fire.)
    """
    cfg = CloudConfig()
    RH = jnp.array([[-0.2, 0.0, 0.5, 1.0, 1.5, 3.0]])
    q_c = jnp.array([[0.0, 1e-2, 1.0, 1e-2, 1.0, 10.0]])       # absurdly large condensate
    q_s = jnp.full((1, 6), 5.0e-3)
    cf_xr = np.asarray(xu_randall_cloud_fraction(RH, q_c, q_s, cfg))
    cf_sq = np.asarray(sundqvist_cloud_fraction(RH, cfg))
    assert np.all((cf_xr >= 0.0) & (cf_xr <= 1.0))
    assert np.all((cf_sq >= 0.0) & (cf_sq <= 1.0))
    assert np.all(np.isfinite(cf_xr)) and np.all(np.isfinite(cf_sq))
    # the [0,1] upper end IS reached (not [0,1)): at RH>=1 with sufficiently large
    # positive q_c the exp underflows to 0 so cf = 1 exactly (indices 3-5 here).
    assert np.any(cf_xr == 1.0)

    RHn = np.asarray(RH)
    # SUNDQVIST clip genuinely fires: un-clipped √-form goes below 0 for RH<rh_crit
    arg = (1.0 - RHn) / max(1.0 - cfg.rh_crit, 1e-6)
    sq_unclipped = np.where(arg > 0.0, 1.0 - np.sqrt(np.where(arg > 0.0, arg, 1.0)), 1.0)
    assert np.any(sq_unclipped < 0.0)
    # XU-RANDALL is bounded to [0,1] by construction (both factors in [0,1]); the
    # un-clipped value never exceeds 1, so its output clip is defensive-only.
    denom = np.maximum((1.0 - RHn) * np.asarray(q_s), 1e-10) ** cfg.gamma_xr
    xr_unclipped = np.clip(RHn, 1e-6, 1.0) ** cfg.p_xr * (
        1.0 - np.exp(-cfg.alpha_xr * np.asarray(q_c) / denom)
    )
    assert np.all((xr_unclipped >= 0.0) & (xr_unclipped < 1.0 + 1e-12))


def test_departure_rh_crit_default_is_tuned_value():
    """Canary: the cloud-fraction diagnostic rh_crit default is the tuned 0.77."""
    assert CloudConfig().rh_crit == 0.77


def test_departure_ice_fraction_linear_ramp():
    """The condensate ice split is a LINEAR ramp 0 (T_freeze) -> 1 (T_ice_only).

    A model choice (not a Xu-Randall/Sundqvist form), pinned at both endpoints,
    the midpoint, and the clipped tails.
    """
    cfg = CloudConfig()
    T_mid = 0.5 * (cfg.T_freeze + cfg.T_ice_only)
    T = jnp.array([[cfg.T_freeze + 5.0, cfg.T_freeze, T_mid, cfg.T_ice_only, cfg.T_ice_only - 5.0]])
    f = np.asarray(_ice_fraction(T, cfg))[0]
    assert f[0] == pytest.approx(0.0)   # above freezing -> clipped to 0
    assert f[1] == pytest.approx(0.0)   # at T_freeze
    assert f[2] == pytest.approx(0.5)   # linear midpoint
    assert f[3] == pytest.approx(1.0)   # at T_ice_only
    assert f[4] == pytest.approx(1.0)   # below T_ice_only -> clipped to 1
