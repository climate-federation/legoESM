"""Scheme-level oracle-faithfulness tests for Thompson hybrid-moment microphysics.

Pins the Thompson et al. (2008) / WRF ``module_mp_thompson.F`` forms that the
JAX scheme reproduces, and locks the documented DEPARTURES as canaries.

FAITHFUL (matched to Thompson-2008 / M2005-lineage):
  * the gamma-distribution shape ratio Γ(μ+4)/Γ(μ+1) = (μ+3)(μ+2)(μ+1);
  * the published vapour-diffusivity fit DV = 8.794e-5·T^1.81/p (prefactor +
    exponent are the Thompson/Reisner constants);
  * the capacitance ice vapour-diffusion growth PRD = η·EPSI·(q_v−q_sat_i)/ABI
    with EPSI = (2π/CONS12^⅓)·ρ·DV·N_i^⅔·q_i_eff^⅓ (q_i_eff = max(clip(q_i,0),
    q_i_min_growth)) — pinned END-TO-END through the public scheme on an isolated
    icy column (the only nonzero vapour sink is ice deposition, so dq_v_dt == −PRD
    to within 1e-9 relative tolerance);
  * Cooper-1986 ice-nucleation constants (N_i0, a, T_act) + the SAM 500/L cap.

DEPARTURE (SURROGATE / extension, locked + labeled):
  * defaults select the faithful capacitance + thompson2008 paths (canary);
  * the "heuristic" ice growth and "bulk_qpower" snow fallbacks are surrogates
    that behaviorally DIFFER from the faithful paths (both live, selectable);
  * warm rain is Seifert-Beheng (gamma-corrected), NOT Thompson warm rain —
    pinned via the γ-correction factors that multiply the SB rates;
  * graupel-from-riming is a smooth-threshold legoESM extension (Part II has no
    graupel category);
  * unknown ice_growth_scheme / snow_scheme raise ValueError (dispatch hardening).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio_ice
from legoesm.atmosphere.physics.microphysics.config import ThompsonConfig
from legoesm.atmosphere.physics.microphysics.thompson import (
    thompson_microphysics,
    _gamma_ratio,
    _DV_PREFACTOR,
    _DV_T_EXP,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState


@pytest.fixture(autouse=True)
def _enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _icy_column(nlev=4, T=250.0, p=4.0e4, super_i=1.05, q_i=1.0e-5, N_i=1.0e8):
    """Single icy, ice-supersaturated column with ONLY q_i > 0.

    q_v is set modestly above ice saturation but below liquid saturation, so the
    ONLY nonzero VAPOUR sink is capacitance ice deposition (no condensation: q_c=0
    & sub-liquid-saturation; no rain evap: q_r=0; no snow deposition: q_s=0). PRCI
    ice->snow and ice sedimentation are q_i-only pathways (never vapour sinks), so
    they don't affect dq_v_dt whether or not they fire (at very low q_i the PRCI
    exp(-LAMI*D_cs) underflows to ~0 anyway). Nucleation is a NUMBER source only
    and is off here (N_i >> N_i_target). Returns the full scheme call signature.
    """
    ncol = 1
    T_a = jnp.full((ncol, nlev), T)
    p_a = jnp.full((ncol, nlev), p)
    qsi = saturation_mixing_ratio_ice(T_a, p_a)
    q_v = qsi * super_i
    q_ia = jnp.full((ncol, nlev), q_i)
    zero = jnp.zeros((ncol, nlev))
    N_ia = jnp.full((ncol, nlev), N_i)
    rho = jnp.full((ncol, nlev), 0.6)
    dz = jnp.full((ncol, nlev), 500.0)
    p_half = jnp.concatenate(
        [p_a[:, :1] * 0.9, 0.5 * (p_a[:, :-1] + p_a[:, 1:]), p_a[:, -1:] * 1.05],
        axis=1,
    )
    hyd = HydrometeorState(
        q_c=zero, q_r=zero, q_i=q_ia, q_s=zero, q_g=zero,
        N_c=jnp.full((ncol, nlev), 1e8), N_r=zero, N_i=N_ia,
    )
    return T_a, q_v, hyd, p_a, p_half, rho, dz, qsi


# ===========================================================================
# FAITHFUL forms
# ===========================================================================
def test_faithful_gamma_ratio_is_the_thompson_shape_ratio():
    """Γ(μ+4)/Γ(μ+1) = (μ+3)(μ+2)(μ+1) — the Thompson-2008 gamma shape ratio.

    Includes fractional μ (the config fields accept non-integer shape parameters),
    where the polynomial identity holds only because Γ(x+3)/Γ(x) = (x+2)(x+1)x is
    exact for real x — so this genuinely certifies the gamma ratio, not just an
    integer factorial coincidence.
    """
    for mu in (0.0, 1.0, 1.5, 2.0, 2.75, 3.0, 5.0, 8.0):
        got = float(_gamma_ratio(mu))
        poly = (mu + 3.0) * (mu + 2.0) * (mu + 1.0)
        gamma = math.gamma(mu + 4.0) / math.gamma(mu + 1.0)
        assert got == pytest.approx(poly, rel=1e-12)
        assert got == pytest.approx(gamma, rel=1e-12)
    # default cloud/rain shape parameters, normalized to the mu=0 baseline
    # _gamma_ratio(0) = 3*2*1 = 6
    cfg = ThompsonConfig()
    assert _gamma_ratio(0.0) == 6.0
    assert _gamma_ratio(cfg.mu_c) / _gamma_ratio(0.0) == pytest.approx(20.0)  # mu_c=3 -> 120/6
    assert _gamma_ratio(cfg.mu_r) / _gamma_ratio(0.0) == pytest.approx(4.0)   # mu_r=1 -> 24/6


def test_faithful_dv_diffusivity_fit_constants():
    """Vapour-diffusivity fit DV = 8.794e-5·T^1.81/p (Thompson/Reisner)."""
    assert _DV_PREFACTOR == 8.794e-5
    assert _DV_T_EXP == 1.81
    # sanity: at 0 C, 1000 hPa the fit gives DV ~ 2.26e-5 m^2/s (textbook value)
    dv = _DV_PREFACTOR * constants.T_freeze ** _DV_T_EXP / 1.0e5
    assert 2.0e-5 < dv < 2.5e-5


def test_faithful_capacitance_prd_pinned_end_to_end():
    """dq_v_dt == −PRD (capacitance form) on an isolated icy column.

    Independent NumPy reimplementation of the Thompson-2008 capacitance vapour-
    diffusion growth pins production to within 1e-9 relative tolerance: on a
    column whose ONLY nonzero vapour sink is ice deposition, the vapour tendency
    equals the negative deposition rate. Certifies EPSI ∝ ρ·DV·N_i^⅔·q_i_eff^⅓
    (q_i_eff = max(clip(q_i,0), q_i_min_growth)), the DV fit, and the ABI
    correction. (PRCI + ice sedimentation are q_i-only pathways, never vapour
    sinks, so they don't enter dq_v_dt regardless of whether they fire.)
    """
    cfg = ThompsonConfig()
    dt = 1.0
    # Two ice loadings: q_i ABOVE the growth floor (q_i_eff == q_i) and q_i BELOW
    # it (q_i_eff == q_i_min_growth), so the max(., q_i_min_growth) floor in EPSI
    # is genuinely exercised and pinned.
    for q_i in (1.0e-5, 0.5 * cfg.q_i_min_growth):
        T_a, q_v, hyd, p_a, p_half, rho, dz, qsi = _icy_column(q_i=q_i)
        out = thompson_microphysics(T_a, q_v, hyd, p_a, p_half, rho, dz, dt, cfg)

        Tn = np.asarray(T_a); pn = np.asarray(p_a); rhon = np.asarray(rho)
        qsi_n = np.asarray(qsi); qv_n = np.asarray(q_v)
        q_i_eff = np.maximum(np.asarray(hyd.q_i), cfg.q_i_min_growth)
        # the floored case must actually hit the floor (q_i_eff > q_i)
        if q_i < cfg.q_i_min_growth:
            assert np.all(q_i_eff == cfg.q_i_min_growth)
        cons12_cbrt = (cfg.rho_cloud_ice * np.pi) ** (1.0 / 3.0)
        dv = _DV_PREFACTOR * Tn ** _DV_T_EXP / np.clip(pn, 1.0, None)
        abi = 1.0 + (constants.L_s * qsi_n / (constants.R_v * Tn ** 2)) * constants.L_s / constants.c_pd
        epsi = 2.0 * np.pi / cons12_cbrt * rhon * dv * np.asarray(hyd.N_i) ** (2.0 / 3.0) * q_i_eff ** (1.0 / 3.0)
        dep_raw = cfg.ice_deposition_efficiency * epsi * (qv_n - qsi_n) / abi
        cap = (qv_n - qsi_n) / max(dt, 1.0)
        assert np.all(dep_raw < cap)                  # uncapped regime (FORM, not the cap)
        prd = np.minimum(np.maximum(dep_raw, 0.0), np.maximum(cap, 0.0))

        dqv = np.asarray(out.dq_v_dt)
        assert np.allclose(dqv, -prd, rtol=1e-9, atol=1e-20)
        # deposition is an ice SOURCE, a vapour SINK, and warms the column (+L_s)
        assert np.all(prd > 0.0)
        assert float(np.max(np.asarray(out.dT_dt))) > 0.0


def test_faithful_cooper_nucleation_constants():
    """Canary: Cooper-1986 ice-nucleation constants + SAM 500/L cap defaults."""
    c = ThompsonConfig()
    assert c.N_i0 == 5.0            # Cooper base number [1/m^3] (0.005/L)
    assert c.cooper_a == 0.304      # Cooper exponent [1/K]
    assert c.cooper_T_act == 265.0  # activation temperature [K]
    assert c.N_i_nuc_max == 5.0e5   # SAM "limit to 500 /L" cap [1/m^3]


# ===========================================================================
# DEPARTURES / surrogate paths
# ===========================================================================
def test_departure_defaults_select_faithful_paths():
    """Canary: defaults select the FAITHFUL capacitance + thompson2008 paths."""
    c = ThompsonConfig()
    assert c.ice_growth_scheme == "capacitance"
    assert c.snow_scheme == "thompson2008"


def test_departure_heuristic_ice_growth_differs_from_capacitance():
    """The legacy 'heuristic' ice growth behaviorally differs from capacitance.

    Compares dq_v_dt (whose ONLY nonzero contribution on this fixture is the
    selected ice-deposition rate — see ``_icy_column``), so the difference is
    attributable specifically to the ice-growth scheme, not to PRCI/sedimentation
    (which enter dq_i_dt). Both outputs are asserted finite.
    """
    dt = 1.0
    T_a, q_v, hyd, p_a, p_half, rho, dz, _ = _icy_column()
    cap = thompson_microphysics(T_a, q_v, hyd, p_a, p_half, rho, dz, dt, ThompsonConfig())
    heur = thompson_microphysics(
        T_a, q_v, hyd, p_a, p_half, rho, dz, dt,
        ThompsonConfig(ice_growth_scheme="heuristic"),
    )
    dqv_cap = np.asarray(cap.dq_v_dt)
    dqv_heur = np.asarray(heur.dq_v_dt)
    assert np.all(np.isfinite(dqv_cap)) and np.all(np.isfinite(dqv_heur))
    assert not np.isclose(float(dqv_cap.ravel()[0]), float(dqv_heur.ravel()[0]), rtol=1e-3)


def test_departure_bulk_qpower_snow_differs_from_thompson2008():
    """'bulk_qpower' snow (surrogate) differs from faithful thompson2008 snow.

    A snowy column at EXACT ice saturation (q_v = q_sat_i) so the Thompson-2008
    snow deposition PRDS ~ 0 for BOTH schemes — the only remaining difference is
    the snow FALL SPEED (faithful Field-2005 mass-weighted vs the capped bulk
    power law), so the different dq_s_dt / surface precipitation is attributable
    solely to sedimentation. Both outputs asserted finite.
    """
    dt = 1.0
    ncol, nlev = 1, 6
    T_a = jnp.full((ncol, nlev), 260.0)
    p_a = jnp.full((ncol, nlev), 5.0e4)
    qsi = saturation_mixing_ratio_ice(T_a, p_a)
    q_v = qsi                                        # exactly saturated: PRDS ~ 0
    q_s = jnp.full((ncol, nlev), 5.0e-4)
    zero = jnp.zeros((ncol, nlev))
    rho = jnp.full((ncol, nlev), 0.7)
    dz = jnp.full((ncol, nlev), 500.0)
    p_half = jnp.concatenate(
        [p_a[:, :1] * 0.9, 0.5 * (p_a[:, :-1] + p_a[:, 1:]), p_a[:, -1:] * 1.05], axis=1,
    )
    hyd = HydrometeorState(
        q_c=zero, q_r=zero, q_i=zero, q_s=q_s, q_g=zero,
        N_c=jnp.full((ncol, nlev), 1e8), N_r=zero, N_i=zero,
    )
    faithful = thompson_microphysics(T_a, q_v, hyd, p_a, p_half, rho, dz, dt, ThompsonConfig())
    bulk = thompson_microphysics(
        T_a, q_v, hyd, p_a, p_half, rho, dz, dt, ThompsonConfig(snow_scheme="bulk_qpower"),
    )
    dqs_f = np.asarray(faithful.dq_s_dt)
    dqs_b = np.asarray(bulk.dq_s_dt)
    assert np.all(np.isfinite(dqs_f)) and np.all(np.isfinite(dqs_b))
    # snow sedimentation profiles differ between the two fall-speed schemes
    assert not np.allclose(dqs_f, dqs_b, rtol=1e-3)
    # surface precipitation also differs (faithful snow falls at a physical speed)
    assert not np.isclose(
        float(np.asarray(faithful.precipitation).ravel()[-1]),
        float(np.asarray(bulk.precipitation).ravel()[-1]),
        rtol=1e-3,
    )


def test_departure_warm_rain_gamma_correction_scales_sb_autoconversion():
    """Warm rain is Seifert-Beheng, gamma-corrected (NOT Thompson warm rain).

    Behavioral pin THROUGH the scheme: on a warm liquid column with q_r = 0 the
    only rain source is SB autoconversion (dq_r_dt == dq_c_au; accretion/evap/melt
    /sedimentation are all zero with q_r = q_i = q_s = q_g = 0). The SB rate is
    dq_c_au = k_au·q_c²·onset·γ_norm·ρ with onset independent of μ, so changing
    ONLY mu_c scales dq_r_dt LINEARLY by γ_c_norm. With mu_c = 3 vs 0 the ratio
    must equal γ_c_norm(3)/γ_c_norm(0) = 20 — certifying both that autoconversion
    is the gamma-corrected SB rate and that the correction is live. A small q_c
    keeps autoconversion below the donor-clamp so the linear factor is visible.
    """
    dt = 1.0
    ncol, nlev = 1, 3
    T_a = jnp.full((ncol, nlev), 290.0)               # warm: f_ice ~ 0
    p_a = jnp.full((ncol, nlev), 8.0e4)
    q_v = jnp.full((ncol, nlev), 0.02)
    q_c = jnp.full((ncol, nlev), 2.0e-4)              # small -> autoconv unclamped
    zero = jnp.zeros((ncol, nlev))
    rho = jnp.full((ncol, nlev), 1.0)
    dz = jnp.full((ncol, nlev), 500.0)
    p_half = jnp.concatenate(
        [p_a[:, :1] * 0.9, 0.5 * (p_a[:, :-1] + p_a[:, 1:]), p_a[:, -1:] * 1.05], axis=1,
    )
    hyd = HydrometeorState(
        q_c=q_c, q_r=zero, q_i=zero, q_s=zero, q_g=zero,
        N_c=jnp.full((ncol, nlev), 2.0e6), N_r=zero, N_i=zero,
    )
    out3 = thompson_microphysics(T_a, q_v, hyd, p_a, p_half, rho, dz, dt, ThompsonConfig(mu_c=3.0))
    out0 = thompson_microphysics(T_a, q_v, hyd, p_a, p_half, rho, dz, dt, ThompsonConfig(mu_c=0.0))
    r3 = float(np.asarray(out3.dq_r_dt).ravel()[0])
    r0 = float(np.asarray(out0.dq_r_dt).ravel()[0])
    assert np.isfinite(r3) and np.isfinite(r0) and r0 > 0.0
    # unclamped: autoconversion is well below the q_c/dt donor cap
    assert r3 < float(np.asarray(q_c).ravel()[0]) / dt
    # linear gamma scaling: ratio == gamma_c_norm(3)/gamma_c_norm(0) == 20
    gamma_ratio = (_gamma_ratio(3.0) / _gamma_ratio(0.0)) / (_gamma_ratio(0.0) / _gamma_ratio(0.0))
    assert gamma_ratio == pytest.approx(20.0)
    assert r3 / r0 == pytest.approx(gamma_ratio, rel=1e-6)


def test_departure_graupel_from_riming_extension_defaults():
    """Canary: the graupel-from-riming SMOOTH-THRESHOLD extension defaults.

    Thompson-2008 Part II has no graupel category; this is a legoESM extension:
    graupel_frac = sigmoid(graupel_sharpness·(riming − rime_to_graupel_threshold))
    times rime_to_graupel_rate.
    """
    c = ThompsonConfig()
    assert c.rime_to_graupel_threshold == 1e-4   # [kg/kg/s]
    assert c.rime_to_graupel_rate == 0.5         # converted fraction [-]
    assert c.graupel_sharpness == 1e4            # sigmoid sharpness


def test_departure_unknown_ice_growth_scheme_raises():
    """Dispatch hardening: an unknown ice_growth_scheme raises ValueError."""
    dt = 1.0
    T_a, q_v, hyd, p_a, p_half, rho, dz, _ = _icy_column()
    with pytest.raises(ValueError, match="ice_growth_scheme"):
        thompson_microphysics(
            T_a, q_v, hyd, p_a, p_half, rho, dz, dt,
            ThompsonConfig(ice_growth_scheme="bogus"),
        )


def test_departure_unknown_snow_scheme_raises():
    """Dispatch hardening: an unknown snow_scheme raises ValueError."""
    dt = 1.0
    T_a, q_v, hyd, p_a, p_half, rho, dz, _ = _icy_column()
    with pytest.raises(ValueError, match="snow_scheme"):
        thompson_microphysics(
            T_a, q_v, hyd, p_a, p_half, rho, dz, dt,
            ThompsonConfig(snow_scheme="bogus"),
        )
