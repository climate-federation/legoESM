"""Oracle-faithfulness tests for the Sundqvist (1978)/SBK89 microphysics.

Oracle: the PUBLISHED equations of Sundqvist (1978) QJRMS 104,677-690 (the
non-convective autoconversion release ``P_auto = c_0*q_c*(1-exp(-(q_c/q_c,cr)^2))``)
and Sundqvist, Berge & Kristjansson (1989) MWR 117,1641-1657 (the Sec. 5 F1
coalescence / F2 Bergeron enhancement STRUCTURE). These pin
``diagnose_sundqvist_process_rates`` against INDEPENDENTLY written expressions of
those forms.

FAITHFUL forms (base autoconversion + the F1/F2 rate-and-threshold enhancement)
are pinned exactly; the DEPARTURES/SURROGATES (F2 Gaussian-in-T proxy, the
simplified condensation and evaporation) are pinned as canaries so a silent
change to the shipped defaults or the surrogate forms trips them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.microphysics.config import SundqvistConfig
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.atmosphere.physics.microphysics.sundqvist import (
    diagnose_sundqvist_process_rates,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    """Numeric faithfulness checks in float64 (restored after)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _hydro(q_c, q_r=None):
    """HydrometeorState with only q_c (and optional q_r); everything else zero."""
    q_c = jnp.asarray(q_c, dtype=jnp.float64)
    z = jnp.zeros_like(q_c)
    q_r = z if q_r is None else jnp.asarray(q_r, dtype=jnp.float64)
    return HydrometeorState(
        q_c=q_c, q_r=q_r, q_i=z, q_s=z, q_g=z, N_c=z, N_r=z, N_i=z,
    )


def _column(T_K, q_c, rh, p_pa=8.0e4, rho_val=1.0, dz_val=500.0, dt=100.0):
    """Build (T, q_v, hydro, p_full, p_half, rho, dz, dt) for target RH.

    RH is set via q_v = rh * q_sat(T, p). Shapes (1, nlev).
    """
    T = jnp.asarray(T_K, dtype=jnp.float64)[None, :]
    q_c = jnp.asarray(q_c, dtype=jnp.float64)[None, :]
    p_full = jnp.full_like(T, p_pa)
    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = jnp.asarray(rh, dtype=jnp.float64)[None, :] * q_sat
    rho = jnp.full_like(T, rho_val)
    dz = jnp.full_like(T, dz_val)
    nlev = T.shape[1]
    p_half = jnp.linspace(p_pa - 1.0e4, p_pa + 1.0e4, nlev + 1)[None, :]
    return T, q_v, _hydro(q_c[0]), p_full, p_half, rho, dz, dt


def _rates(cfg, T_K, q_c, rh, **kw):
    T, q_v, hydro, p_full, p_half, rho, dz, dt = _column(T_K, q_c, rh, **kw)
    return diagnose_sundqvist_process_rates(
        T=T, q_v=q_v, hydrometeors=hydro, p_full=p_full, p_half=p_half,
        rho=rho, dz=dz, dt=dt, config=cfg,
    )


# ===========================================================================
# FAITHFUL: the published autoconversion forms
# ===========================================================================
def test_faithful_base_autoconversion_sundqvist1978():
    """P_auto = c_0*q_c*(1-exp(-(q_c/q_c,crit)^2)) with F1=F2=1 (enh off)."""
    cfg = SundqvistConfig(coalescence_enh_coeff=0.0, bergeron_enh_coeff=0.0,
                          evap_coeff=0.0)
    # Warm (F2 off anyway), sub-critical RH -> no condensation/evaporation.
    q_c = np.array([2e-4, 5e-4, 1e-3])
    T = np.full(3, 290.0)
    r = _rates(cfg, T, q_c, rh=np.full(3, 0.5))
    expected = cfg.auto_rate * q_c * (1.0 - np.exp(-(q_c / cfg.qc_crit) ** 2))
    # donor cap c_0*... <= q_c/dt holds for these values (checked): direct pin.
    assert np.allclose(np.asarray(r.autoconversion)[0], expected, rtol=1e-9, atol=1e-16)


def test_faithful_sbk89_enhancement_multiplies_rate_and_lowers_threshold():
    """At the TOP level P_above=0 => F1=1, enh=F2; pin the enh in BOTH the rate
    prefactor AND the (q_c*enh/q_c,crit) threshold (SBK89 Sec. 5 structure)."""
    cfg = SundqvistConfig(evap_coeff=0.0)  # coalescence + bergeron ON (defaults)
    T_peak = float(cfg.bergeron_T_peak_K)
    q_c = np.array([8e-4])
    r = _rates(cfg, np.array([T_peak]), q_c, rh=np.array([0.5]))
    f2 = 1.0 + cfg.bergeron_enh_coeff * np.exp(
        -((T_peak - cfg.bergeron_T_peak_K) / cfg.bergeron_T_width_K) ** 2
    )  # = 1 + c2 at the peak
    enh = 1.0 * f2  # coal F1 = 1 at the top level
    expected = cfg.auto_rate * enh * q_c * (
        1.0 - np.exp(-(q_c * enh / cfg.qc_crit) ** 2)
    )
    assert np.allclose(np.asarray(r.autoconversion)[0, 0], expected[0], rtol=1e-9, atol=1e-16)


def test_faithful_f1_coalescence_uses_precip_from_above():
    """F1 = 1 + c1*sqrt(P_above): level below sees enhanced release from the
    autoconversion precip flux produced above (SBK89 collection form)."""
    cfg = SundqvistConfig(bergeron_enh_coeff=0.0, evap_coeff=0.0)  # isolate F1
    q_c = np.array([1e-3, 1e-3])  # 2 levels, identical cloud water
    rho_val, dz_val = 1.0, 500.0
    r = _rates(cfg, np.full(2, 290.0), q_c, rh=np.full(2, 0.5),
               rho_val=rho_val, dz_val=dz_val)
    auto = np.asarray(r.autoconversion)[0]

    # Level 0 (top): base form; its precip flux enters level 1 from above.
    a0 = cfg.auto_rate * q_c[0] * (1.0 - np.exp(-(q_c[0] / cfg.qc_crit) ** 2))
    P0 = a0 * rho_val * dz_val
    f1 = 1.0 + cfg.coalescence_enh_coeff * np.sqrt(P0)
    a1 = cfg.auto_rate * f1 * q_c[1] * (1.0 - np.exp(-(q_c[1] * f1 / cfg.qc_crit) ** 2))
    assert np.allclose(auto[0], a0, rtol=1e-9, atol=1e-16)
    assert np.allclose(auto[1], a1, rtol=1e-9, atol=1e-16)
    assert auto[1] > auto[0]  # F1>1 enhances the lower level


# ===========================================================================
# DEPARTURES / SURROGATES: forms that differ from the SBK89 closed equations
# ===========================================================================
def test_departure_f2_is_gaussian_in_T_not_saturation_difference():
    """F2 = 1 + c2*exp(-((T-T_peak)/T_width)^2): peaks at T_peak, ->1 warm.

    Pins the SURROGATE Gaussian form (NOT SBK89's e_sw-e_si driver). Isolated at
    the top level with coalescence off so enh == F2.
    """
    cfg = SundqvistConfig(coalescence_enh_coeff=0.0, evap_coeff=0.0)
    q_c_val = 8e-4
    for T_K in (float(cfg.bergeron_T_peak_K), 300.0, float(cfg.bergeron_T_peak_K) - 20.0):
        r = _rates(cfg, np.array([T_K]), np.array([q_c_val]), rh=np.array([0.5]))
        f2 = 1.0 + cfg.bergeron_enh_coeff * np.exp(
            -((T_K - cfg.bergeron_T_peak_K) / cfg.bergeron_T_width_K) ** 2
        )
        expected = cfg.auto_rate * f2 * q_c_val * (
            1.0 - np.exp(-(q_c_val * f2 / cfg.qc_crit) ** 2)
        )
        assert np.allclose(np.asarray(r.autoconversion)[0, 0], expected, rtol=1e-9, atol=1e-16)
    # Warm limit -> F2 ~ 1 (no mixed-phase enhancement).
    warm = _rates(cfg, np.array([305.0]), np.array([q_c_val]), rh=np.array([0.5]))
    base = cfg.auto_rate * q_c_val * (1.0 - np.exp(-(q_c_val / cfg.qc_crit) ** 2))
    assert np.allclose(np.asarray(warm.autoconversion)[0, 0], base, rtol=1e-6, atol=1e-16)


def test_departure_condensation_removes_supersaturation_toward_qsat():
    """SURROGATE condensation FORM: sigmoid(k*(RH-RH_crit))*max(q_v-q_sat,0)/dt.

    This pins the scheme's SIMPLIFIED condensation form (a departure from SBK89's
    full condensation closure), NOT the saturation curve: q_sat is taken from the
    model's own ``saturation_mixing_ratio`` in both the fixture and the expected
    value, so this deliberately does not certify the saturation routine (that is
    ``thermo``'s oracle, tested separately) — only that the scheme applies the
    sigmoid RH gate to the supersaturation ``q_v-q_sat``.

    At RH>1 (supersaturated) it removes q_v-q_sat (drives toward q_sat); at
    RH_crit<RH<1 (unsaturated) it is ZERO -- the documented fix (does NOT strip
    vapor down to RH_crit*q_sat).
    """
    cfg = SundqvistConfig()
    dt = 100.0
    T = np.array([290.0])
    p = 8.0e4
    q_sat = float(saturation_mixing_ratio(jnp.array([[290.0]]), jnp.full((1, 1), p))[0, 0])

    # Supersaturated RH=1.2 -> remove q_v - q_sat with a near-1 sigmoid gate.
    r_sup = _rates(cfg, T, np.array([1e-4]), rh=np.array([1.2]), p_pa=p, dt=dt)
    rh = 1.2
    gate = 1.0 / (1.0 + np.exp(-cfg.sigmoid_sharpness * (rh - cfg.rh_crit)))
    expected = gate * (rh * q_sat - q_sat) / dt
    assert np.allclose(float(np.asarray(r_sup.condensation)[0, 0]), expected, rtol=1e-9, atol=1e-16)

    # Unsaturated (RH_crit<RH=0.9<1): no supersaturation -> condensation == 0.
    r_unsat = _rates(cfg, T, np.array([1e-4]), rh=np.array([0.9]), p_pa=p, dt=dt)
    assert float(np.asarray(r_unsat.condensation)[0, 0]) == 0.0


def test_departure_default_constants():
    """Shipped ``SundqvistConfig`` defaults (canary): a silent change trips this.

    These are the model's tuned closure defaults, NOT certified SBK89 paper
    constants: c1/c2 are re-tunable and F2's amplitude/peak/width parameterize the
    Gaussian surrogate (not SBK89's e_sw-e_si driver).
    """
    c = SundqvistConfig()
    assert c.rh_crit == 0.8
    assert c.auto_rate == 1e-3         # c_0 [1/s]
    assert c.qc_crit == 5e-4           # q_c,crit [kg/kg]
    assert c.coalescence_enh_coeff == 300.0   # c1 (SBK89 F1)
    assert c.bergeron_enh_coeff == 3.0        # c2 (F2 Gaussian amplitude)
    assert c.bergeron_T_peak_K == constants.T_freeze - 15.0
    assert c.bergeron_T_width_K == 7.0
    assert c.evap_coeff == 5e-4
