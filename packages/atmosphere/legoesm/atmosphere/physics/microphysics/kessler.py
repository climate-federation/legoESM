"""Kessler warm-rain microphysics column backend.

A simple one-moment warm-rain scheme tracking cloud water and rain.
Processes: saturation adjustment, autoconversion, accretion, evaporation,
rain sedimentation, and latent heating.

Operations use AD-safe approximations (smooth switches + guarded fractional
powers) for jax.grad compatibility; the scheme is NOT globally smooth — it
retains max/min/clip kinks (donor clamps, positivity floors).

Faithfulness to Kessler (1969)
------------------------------
FAITHFUL (classic Kessler process forms + coefficients):
  * **Autoconversion** cloud->rain ``A = k1·max(q_c − a, 0)`` with the canonical
    Kessler / Klemp-Wilhelmson constants ``k1 = 1e-3 s⁻¹`` (``autoconversion_rate``)
    and threshold ``a = 1e-3 kg/kg`` (``autoconversion_threshold``).
  * **Accretion** (collection of cloud by rain) ``C = k2·q_c·q_r^0.875`` with
    ``k2 = 2.2`` (``accretion_coeff``) and the classic 0.875 exponent.
  * **Sedimentation density correction** ``(ρ_sfc/max(ρ, 0.1))^0.5`` — the classic
    ``(ρ_0/ρ)^½`` air-density fall-speed factor (ρ floored at 0.1 kg/m³ aloft for
    AD safety).
  * **Latent heating** ``dT/dt = L_v·(cond − evap)/c_pd`` — thermodynamically
    consistent with the vapour<->liquid exchange.
  * **Rain evaporation** — Klemp & Wilhelmson (1978) eq. 2.15, in the form coded
    in WRF ``phys/module_mp_kessler.F``::

        E = (1.6 + 124.9·(ρ' q_r)^0.2046)·(ρ' q_r)^0.525
            / (2.55e8/(p q_sat) + 5.4e5) · (q_sat − q_v)₊/(ρ' q_sat)

    with ``ρ' = 10⁻³ ρ`` the density in g cm⁻³ and ``p`` in Pa (WRF:
    ``pressure = 1.e5*pii**(1004/287)``; KW78's ``2.55e6/(p q_vs)`` with p in mb
    is the same term), times
    ``evaporation_coeff`` (1 = the published rate). Kept from the reference: the
    cap that rain cannot evaporate more than the sub-saturation left after the
    cloud water has evaporated (WRF ``max(-prod - qc, 0)``) and the ``E·dt ≤ q_r``
    donor cap. ``q_sat`` is the scheme's own saturation mixing ratio, not WRF's
    Tetens fit.
DEPARTURES / SURROGATES:
  * **Constant rain fall speed**: ``V_t = rain_fall_speed·(ρ_sfc/max(ρ, 0.1))^0.5``
    with a configurable DEFAULT ``rain_fall_speed = 5 m/s``.  The classic Kessler mass-
    weighted terminal velocity is q_r-DEPENDENT (``∝ (ρ q_r)^0.1364``); dropping
    that dependence makes the sedimentation flux LINEAR in q_r (not ``q_r^1.1364``).
  * **Smooth saturation adjustment**: a sigmoid switch (``saturation_sharpness``)
    replaces Kessler's hard/instantaneous saturation adjustment, for
    differentiability.
  * **AD-safe / differentiable positivity limiters**: positivity limiting itself is
    standard in Kessler implementations, but the SPECIFIC differentiable forms here
    are departures — ``safe_pow`` fractional-power guards, a condensation donor
    clamp against q_v, a joint q_c-sink donor clamp, an RH-deficit resolution floor,
    and a dt-limited sedimentation flux.  None alter the classic rates in the
    unclamped, resolved regime.
Non-behavioral pins: ``tests/atmosphere/hydrostatic/unit/test_kessler_faithful.py``.

CloudBench parameter set (``KesslerConfig.parameter_set="cloudbench"``)
-----------------------------------------------------------------------
A second, selectable set of warm-rain RATES on the same scheme (same saturation
adjustment, donor clamps, sedimentation operator and latent heating): the
CloudBench LES one-moment warm rain as coded in swirl-lm -- threshold
autoconversion, CloudMicrophysics-1M accretion and rain evaporation with the
Azimi et al. (2024) EKI coefficients, and the Chen et al. (2022) rain fall speed
(with swirl-lm's radius-for-diameter substitution reproduced on purpose).
Formulas, values and every source line are in the constants block below; the
classic set stays the default. Pins: ``tests/atmosphere/hydrostatic/unit/
test_kessler_cloudbench.py``.

References
----------
- Kessler (1969): On the Distribution and Continuity of Water
  Substance in Atmospheric Circulations.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import gamma as _jgamma

from legoesm import constants
from legoesm.thermo import saturation_vapor_pressure
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    safe_pow,
    donor_clamp_scale,
    saturation_adjustment,
    saturation_deficit_rate,
)
from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    sedimentation_tendency,
)


# Kessler accretion exponent + density floor (fixed).
_KESSLER_ACCR_EXP = 0.875
_RHO_FLOOR = 0.1

# --- rain evaporation (Klemp & Wilhelmson 1978, J. Atmos. Sci. 35, eq. 2.15;
# constants as coded in WRF phys/module_mp_kessler.F, pressure in Pa) ---
_KW78_VENT_A = 1.6               # ventilation polynomial constant [-]
_KW78_VENT_B = 124.9             # ventilation polynomial coefficient [-]
_KW78_VENT_EXP = 0.2046          # ventilation exponent on ρ' q_r
_KW78_CONTENT_EXP = 0.525        # rain-content exponent (Marshall-Palmer)
_KW78_RESIST_CONST = 5.4e5       # thermal-conduction resistance term
_KW78_RESIST_P = 2.55e8          # vapour-diffusion resistance numerator (p in Pa)
_KG_M3_TO_G_CM3 = 1.0e-3         # ρ [kg m⁻³] -> ρ' [g cm⁻³] (the KW78 cgs content)
_QSAT_FLOOR = 1.0e-10            # division guard on q_sat [kg/kg]

# =============================================================================
# CloudBench one-moment warm-rain set (``KesslerConfig.parameter_set="cloudbench"``)
# =============================================================================
# CloudBench (Chammas et al. 2026, Sci. Adv. doi 10.1126/sciadv.aec8488, Methods
# "Microphysics") cites CloudMicrophysics.jl and Azimi et al. (2024) for the warm
# rain and Chen et al. (2022) for the fall speed. The FORMS below reproduce the
# public swirl-lm one-moment code (github.com/google-research/swirl-lm, commit
# 98f96f801d71b9e303f22c51e615f41b29ea64c4, swirl_lm/physics/atmosphere/), the LES
# the paper describes. NOT VERIFIED to be the exact code version that produced the
# CloudBench data: the run configs (config.pbtxt) sit on an internal path and are
# not in the public bucket. User decisions 2026-10-19 fix every choice below.
#   * Autoconversion: swirl-lm ``autoconversion_and_accretion`` builds a DEFAULT
#     ``Autoconversion()`` proto (microphysics_one_moment.py L594-597), so the run
#     uses ``max(q_c - q_thr, 0)/tau`` (L319) with the proto defaults, whatever
#     the config says -- reproduced, NOT Azimi's no-threshold q_c/tau_0 form.
#   * Accretion (L263-296), rain evaporation (L453-477, ventilation L395-406,
#     conduction/diffusion L221-226) and the Marshall-Palmer lambda (particles.py
#     L189-202): CloudMicrophysics 1M power-law forms (Azimi 2024 eqs. 7, 8, 10,
#     12, 13) with the Azimi 2024 Table 3 EKI terminal-velocity and accretion
#     coefficients (chi_v, Delta_v, chi_a, Delta_a) and b_vent.
#   * Rain fall speed: Chen et al. (2022) eq. 20, Table B1
#     (terminal_velocity_chen2022.py L220-233, L264-312, L345-363).
#     PROBABLE CLOUDBENCH BUG, REPRODUCED DELIBERATELY FOR COMPARABILITY: Chen's
#     fit is in DIAMETER, but swirl-lm passes the RADIUS-based Marshall-Palmer
#     lambda (dN/dr = n0 exp(-lambda r)) unchanged, i.e. treats r as D.
#     CloudMicrophysics.jl (v0.46.0, Microphysics1M.jl ``terminal_velocity``
#     for Chen2022VelTypeRain) converts with lambda_D = lambda_r/2. The swirl-lm
#     form gives ~2.1 m/s at q_r = 0.1 g/kg, rho = 1.1; CloudMicrophysics ~4.0.
#   * Rates are applied unchanged to this scheme's mixing ratios (swirl-lm uses
#     specific humidities); latent heat is this model's constant ``L_v``.
#   * Saturation in the evaporation (S = q_v/q_sat and p_sat in G(T)) is this
#     model's own curve (``legoesm.thermo``, the same q_sat the scheme's
#     condensation uses), NOT swirl-lm's integrated Clausius-Clapeyron
#     (thermodynamics/water.py L748, q_sat = p_sat/(rho R_v T) L845-848): a second
#     saturation curve inside one scheme would break its thermodynamic
#     consistency and is barred by the repo's no-saturation-re-implementation rule;
#     saturation is outside the approved scope (warm-rain constants + fall speed
#     only). Measured: swirl-lm integrates Clausius-Clapeyron from the triple point
#     with Kirchhoff L(T) = lh_v0 + (cp_v - cp_l)(T - t_0) (water.py L162, L748-751),
#     p_sat = 611.7 (T/273.16)^(dcp/R_v) exp((lh_v0 - dcp t_0)/R_v (1/273.16 - 1/T)),
#     dcp = 1859.0 - 4219.9, R_v = 461.89, and its default lh_v0 = 2.258e6 J/kg (the
#     100 C value) at t_0 = 273 K (thermodynamics.proto L88-141). That p_sat is
#     10.8 % / 16.1 % below Murphy & Koop (2005) at 290 / 300 K (exact at 273.16 K;
#     with lh_v0 = 2.501e6 it is within 0.33 %); this model's curve is within 0.11 %.
#   * Cloud-droplet settling (swirl-lm sediments cloud liquid) is NOT included;
#     the fixed droplet number 1e8 m^-3 (microphysics_one_moment_constants.py L32;
#     SwirlLMCloudBench.jl 7d5c9c16 src/les_config.jl L49) feeds only that settling
#     and the radiation effective radius in swirl-lm, so it has no consumer here.
# --- autoconversion (swirl-lm microphysics.proto, OneMoment.Autoconversion) ---
_CB_AUTOCONV_TAU_S = 1.0e3       # tau_lr [s], proto L109
_CB_AUTOCONV_Q_THR = 5.0e-4      # q_l_threshold [kg/kg], proto L113
# --- rain particle (proto OneMoment.Rain; = ClimaParams v1.3.3 defaults) ---
_CB_RHO_RAIN = 1.0e3             # rho [kg/m^3], proto L33 (= constants.rho_water)
_CB_DRAG_COEFF = 0.55            # c_d, proto L35 (ClimaParams rain_drop_drag_coefficient)
_CB_R0 = 1.0e-3                  # r_0 [m], proto L37
_CB_M_E = 3.0                    # m_e, proto L39
_CB_CHI_M = 1.0                  # chi_m, proto L41
_CB_DEL_M = 0.0                  # del_m, proto L42
_CB_A_E = 2.0                    # a_e, proto L44
_CB_V_E = 0.5                    # v_e, proto L48
_CB_A_VENT = 1.5                 # a_vent, proto L53 (Azimi Table 2, not calibrated)
_CB_N0 = 1.6e7                   # n_0 [m^-4], proto L56 (ClimaParams ..._n0)
_CB_E_LR = 0.8                   # cloud-rain collision efficiency, proto L123
# --- Azimi et al. (2024, JAMES 16, e2023MS004028) Table 3, EKI optimal values ---
_CB_CHI_V = 0.205                # terminal-velocity coefficient chi_v
_CB_DEL_V = 0.228                # terminal-velocity coefficient Delta_v
_CB_CHI_A = 16.61                # accretion coefficient chi_a
_CB_DEL_A = 3.00                 # accretion coefficient Delta_a
_CB_B_VENT = 0.98                # ventilation coefficient b_vent
# --- air properties (swirl-lm microphysics_one_moment_constants.py) ---
_CB_K_COND = 2.4e-2              # thermal conductivity of air [J/(m s K)], L24
_CB_NU_AIR = 1.6e-5              # kinematic viscosity of air [m^2/s], L26
_CB_D_VAP = 2.26e-5              # water-vapour diffusivity [m^2/s], L28
# --- Chen et al. (2022, Atmos. Res. 273, 106171) Table B1, rain; swirl-lm
# terminal_velocity_chen2022_tables.py L23-31 = ClimaParams v1.3.3
# parameters.toml L1884-1912 (Chen2022_table_B1_*) ---
_CHEN_B1_A = (0.044612, -0.263166, 4.7178)   # a_i [mm^-b_i]
_CHEN_B1_B = (2.2955, 2.2955, 1.1451)        # b_i at rho = 0 [-]
_CHEN_B1_B_RHO = 0.038465                    # b_i = b_i0 - b_rho * rho [m^3/kg]
_CHEN_B1_C = (0.0, 0.184325, 0.184325)       # c_i [mm^-1]
_CHEN_B1_A3_POW = -0.47335                   # a_3 *= rho^a3_pow
_CHEN_B1_Q = 0.115231                        # a_i *= exp(q * rho) [m^3/kg]
_MM_PER_M = 1.0e3                            # Chen's mm -> SI m
_CHEN_MOMENT_DELTA = 4.0                     # mu + k + 1, mu = 0 (exponential), k = 3

# Derived (fixed) combinations.
# Marshall-Palmer: lambda^(m) = Gamma(m) chi_m m0 n0 / (q rho r0^(m_e+del_m)),
# m = m_e + del_m + 1  (particles.py L194-202); write lambda^-1 = (rho q / _CB_LAM_NUM)^(1/m).
_CB_M = _CB_M_E + _CB_DEL_M + 1.0
_CB_M0 = 4.0 / 3.0 * math.pi * _CB_RHO_RAIN * _CB_R0 ** 3
_CB_LAM_NUM = (math.gamma(_CB_M) * _CB_CHI_M * _CB_M0 * _CB_N0
               / _CB_R0 ** (_CB_M_E + _CB_DEL_M))
_CB_A0 = math.pi * _CB_R0 ** 2
_CB_SIGMA_AV = _CB_A_E + _CB_V_E + _CB_DEL_A + _CB_DEL_V
_CB_VENT_EXP = 0.5 * (_CB_V_E + _CB_DEL_V)
# Gamma((v_e + Delta_v + 5)/2) argument = power of 1/lambda in the ventilated
# evaporation term (lambda^-2 * lambda^-(v_e+Delta_v)/2 * lambda^-1/2).
_CB_VENT_GAMMA_ARG = _CB_VENT_EXP + 2.5
# v_0^2 = _CB_V0_SQ_COEFF * g * (rho_w/rho - 1)   (Azimi 2024 eq. 7)
_CB_V0_SQ_COEFF = 8.0 * _CB_R0 / (3.0 * _CB_DRAG_COEFF)
_KESSLER_PARAMETER_SETS = ("classic", "cloudbench")

__physics_contract__ = {
    "summary": (
        "Kessler (1969) single-moment warm-rain microphysics: saturation "
        "adjustment (cloud condensation/evaporation), cloud->rain "
        "autoconversion, accretion, rain evaporation and rain sedimentation to "
        "surface precipitation."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "hydrometeors.q_c": "kg/kg",
        "hydrometeors.q_r": "kg/kg", "p_full": "Pa", "rho": "kg/m^3",
        "dz": "m", "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_dt": "kg/kg/s",
        "dq_r_dt": "kg/kg/s", "precipitation": "kg/m^2/s",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. Latent heating dT_dt is consistent with the "
        "vapour<->liquid phase-change rate via L_v (condensation warms, "
        "evaporation cools). Water is redistributed vapour<->cloud<->rain; "
        "SURFACE PRECIPITATION (>= 0) removes water from the column, so column "
        "moisture is NOT conserved and the falling condensate carries enthalpy "
        "out -- no contract-level conservation is claimed. Mixing ratios >= 0."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "Kessler (1969), Meteorol. Monogr. 10(32), Amer. Meteorol. Soc.",
    "idealized_test": (
        "tests/unit/test_physics_microphysics.py — supersaturated air condenses "
        "with L_v heating; cloud above the autoconversion threshold makes rain "
        "+ surface precip; subsaturated rain evaporates; mixing ratios stay "
        ">= 0; differentiable wrt T, q_v."
    ),
}


def kw78_rain_evaporation(q_v, q_r, q_c, q_sat, p, rho, demand, dt, scale=1.0):
    """Klemp & Wilhelmson (1978) rain evaporation rate [kg/kg/s] (see module doc).

    ``demand`` is the scheme's signed saturation-adjustment rate WITHOUT the
    cloud-water clamp (negative = sub-saturation); rain may evaporate only the
    part left after all cloud water has evaporated, ``max(−demand − q_c/dt, 0)``
    (WRF ``max(-prod - qc, 0)``), and at most ``q_r/dt``."""
    q_r_pos = jnp.clip(q_r, 0.0, None)
    q_sat_s = jnp.clip(q_sat, _QSAT_FLOOR)
    rho_cgs = rho * _KG_M3_TO_G_CM3
    content = rho_cgs * q_r_pos
    vent = _KW78_VENT_A + _KW78_VENT_B * safe_pow(content, _KW78_VENT_EXP)
    resist = _KW78_RESIST_P / (p * q_sat_s) + _KW78_RESIST_CONST
    rate = (scale * vent * safe_pow(content, _KW78_CONTENT_EXP) / resist
            * jnp.clip(q_sat - q_v, 0.0) / (rho_cgs * q_sat_s))
    return _cap_rain_evaporation(rate, q_r, q_c, demand, dt)


def _cap_rain_evaporation(rate, q_r, q_c, demand, dt):
    """Rain may evaporate only the sub-saturation left after all cloud water has
    evaporated, ``max(-demand - q_c/dt, 0)`` (WRF ``max(-prod - qc, 0)``), and at
    most ``q_r/dt``. Scales a rate only, so water and enthalpy still close."""
    dt_s = jnp.maximum(dt, 1.0e-12)
    rate = jnp.minimum(rate, jnp.clip(-demand - jnp.clip(q_c, 0.0) / dt_s, 0.0))
    return jnp.minimum(rate, jnp.clip(q_r, 0.0, None) / dt_s)


# --- CloudBench one-moment set (see the constants block for provenance) ---

def _cb_inv_lambda_pow(q_r, rho, power):
    """``lambda**-power`` of the Marshall-Palmer rain distribution (radius-based,
    dN/dr = n0 exp(-lambda r)), AD-safe at ``q_r = 0`` (value 0, finite grad)."""
    x = jnp.clip(rho, _RHO_FLOOR) * jnp.clip(q_r, 0.0, None) / _CB_LAM_NUM
    return safe_pow(x, power / _CB_M)


def _cb_unit_fall_speed(rho):
    """Reference raindrop speed v_0 = sqrt(8 r0 g (rho_w/rho - 1) / (3 C_d)) [m/s]
    (Azimi 2024 eq. 7; swirl-lm ``_v_0``)."""
    rho_s = jnp.clip(rho, _RHO_FLOOR)
    return jnp.sqrt(_CB_V0_SQ_COEFF * constants.g * (_CB_RHO_RAIN / rho_s - 1.0))


def cloudbench_autoconversion(q_c):
    """Cloud -> rain autoconversion ``max(q_c - q_thr, 0)/tau`` [kg/kg/s] with the
    swirl-lm proto defaults (tau = 1000 s, q_thr = 5e-4 kg/kg)."""
    return jnp.maximum(q_c - _CB_AUTOCONV_Q_THR, 0.0) / _CB_AUTOCONV_TAU_S


def cloudbench_accretion(q_c, q_r, rho):
    """Collection of cloud by rain [kg/kg/s] (Azimi 2024 eq. 10):
    ``n0 a0 chi_a chi_v v0 E q_c Gamma(S+1) / lambda / (r0 lambda)^S``,
    ``S = a_e + v_e + Delta_a + Delta_v``."""
    coeff = (_CB_N0 * _CB_A0 * _CB_CHI_A * _CB_CHI_V * _CB_E_LR
             * math.gamma(_CB_SIGMA_AV + 1.0) / _CB_R0 ** _CB_SIGMA_AV)
    return (coeff * _cb_unit_fall_speed(rho) * q_c
            * _cb_inv_lambda_pow(q_r, rho, _CB_SIGMA_AV + 1.0))


def cloudbench_rain_evaporation_rate(T, q_v, q_r, q_sat, rho):
    """Uncapped rain evaporation [kg/kg/s, > 0 removes rain] (Azimi 2024 eqs. 12,
    13; swirl-lm ``evaporation_sublimation`` for rain, zero when S >= 1):
    ``4 pi n0/rho (1 - S) G(T) lambda^-2 [a_vent + b_vent (nu/D)^(1/3)
    (r0 lambda)^-((v_e+Delta_v)/2) (2 chi_v v0/(nu lambda))^(1/2)
    Gamma((v_e+Delta_v+5)/2)]`` with ``S = q_v/q_sat`` and this model's constant L_v."""
    rho_s = jnp.clip(rho, _RHO_FLOOR)
    q_sat_s = jnp.clip(q_sat, _QSAT_FLOOR)
    subsat = jnp.clip(q_sat - q_v, 0.0) / q_sat_s                  # (1 - S)+
    lv = constants.L_v
    rvt = constants.R_v * T
    g_cd = 1.0 / (lv / (_CB_K_COND * T) * (lv / rvt - 1.0)
                  + rvt / (saturation_vapor_pressure(T) * _CB_D_VAP))
    vent_b = (_CB_B_VENT * (_CB_NU_AIR / _CB_D_VAP) ** (1.0 / 3.0)
              / _CB_R0 ** _CB_VENT_EXP
              * jnp.sqrt(2.0 * _CB_CHI_V * _cb_unit_fall_speed(rho) / _CB_NU_AIR)
              * math.gamma(_CB_VENT_GAMMA_ARG))
    bracket = (_CB_A_VENT * _cb_inv_lambda_pow(q_r, rho, 2.0)
               + vent_b * _cb_inv_lambda_pow(q_r, rho, _CB_VENT_GAMMA_ARG))
    return 4.0 * math.pi * _CB_N0 / rho_s * subsat * g_cd * bracket


def chen2022_rain_fall_speed(q_r, rho):
    """Mass-weighted rain fall speed [m/s], Chen et al. (2022) eq. 20 with Table B1
    over the exponential distribution (mu = 0, k = 3):
    ``sum_i a_i Gamma(b_i+4)/Gamma(4) lambda^-b_i / (1 + c_i/lambda)^(b_i+4)``,
    clipped at 0 (swirl-lm). DELIBERATE CloudBench reproduction: ``lambda`` is the
    RADIUS-based Marshall-Palmer parameter used where Chen's fit expects the
    diameter one (probable swirl-lm bug, see the constants block)."""
    rho_s = jnp.clip(rho, _RHO_FLOOR)
    q_fac = jnp.exp(_CHEN_B1_Q * rho_s)
    a_rho = (_CHEN_B1_A[0] * q_fac, _CHEN_B1_A[1] * q_fac,
             _CHEN_B1_A[2] * q_fac * rho_s ** _CHEN_B1_A3_POW)
    inv_lam = _cb_inv_lambda_pow(q_r, rho, 1.0)
    v = jnp.zeros_like(inv_lam)
    for a_i, b_i0, c_i in zip(a_rho, _CHEN_B1_B, _CHEN_B1_C):
        b_i = b_i0 - _CHEN_B1_B_RHO * rho_s
        a_si = a_i * _MM_PER_M ** b_i                 # mm^-b -> m^-b
        c_si = c_i * _MM_PER_M                        # mm^-1 -> m^-1
        v = v + (a_si * _jgamma(b_i + _CHEN_MOMENT_DELTA)
                 / math.gamma(_CHEN_MOMENT_DELTA)
                 * _cb_inv_lambda_pow(q_r, rho, b_i)
                 / (1.0 + c_si * inv_lam) ** (b_i + _CHEN_MOMENT_DELTA))
    return jnp.maximum(v, 0.0)


def kessler_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: KesslerConfig = KesslerConfig(),
) -> MicrophysicsOutput:
    """Compute Kessler microphysics tendencies.

    Parameters
    ----------
    T : jax.Array
        Temperature [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    hydrometeors : HydrometeorState
        Hydrometeor state (only q_c, q_r used).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    rho : jax.Array
        Air density [kg/m^3], shape (ncol, nlev).
    dz : jax.Array
        Layer thickness [m], shape (ncol, nlev).
    dt : float
        Time step [s].
    config : KesslerConfig

    Returns
    -------
    MicrophysicsOutput
    """
    if config.parameter_set not in _KESSLER_PARAMETER_SETS:
        raise ValueError(
            f"unknown Kessler microphysics parameter_set "
            f"{config.parameter_set!r}; expected one of {_KESSLER_PARAMETER_SETS}")
    cloudbench = config.parameter_set == "cloudbench"
    ncol, nlev = T.shape
    q_c = hydrometeors.q_c
    q_r = hydrometeors.q_r
    sharpness = config.saturation_sharpness

    # 1. Saturation adjustment — the SHARED ``_warm_rain.saturation_adjustment``
    # helper computes ``q_sat`` and the supersaturation residual (in fp64 when
    # x64 is available — the issue #618 robustness path the fp32 LES/CRM
    # production config relies on; an inline fp32 copy here re-exposed the
    # ~65 % LWP deficit #618 fixed), applies the psychrometric correction, and
    # donor-clamps the evaporation branch against ``q_c``.  Kessler EXTENDS it
    # with the previously MISSING positive-branch bound: condensation is
    # donor-clamped by the available ``q_v``.  Without that clamp the coupled
    # driver floors ``q_v`` independently while keeping the full ``q_c``
    # increment, so saturation adjustment created cloud water from vapour that
    # was floored away → ``q_c`` accumulated to physically impossible
    # ~1 kg/kg (opaque clouds, planetary-albedo runaway, the coupled cold
    # drift / OLR collapse).
    _dt_safe = jnp.maximum(dt, 1e-10)
    condensation, q_sat = saturation_adjustment(
        T, q_v, p_full, _dt_safe, sharpness, q_c=q_c,
        hard_adjust=config.hard_saturation_adjustment,
        hard_threshold=config.hard_sat_adjust_threshold,
        hard_max_heating_K=config.hard_sat_max_heating_K,
    )
    q_v_avail = jnp.clip(q_v, 0.0, None)
    condensation = jnp.minimum(condensation, q_v_avail / _dt_safe)

    dq_v_sat = -condensation
    dq_c_sat = condensation

    # 2. Autoconversion: cloud -> rain (threshold excess).  Uses raw
    # ``q_c`` so the joint donor clamp below scales it consistently
    # with the other ``q_c`` sinks (an earlier form used
    # ``q_c_updated = q_c + cond·dt`` here, which double-counted the
    # condensation budget when joint sinks were active and made the
    # joint clamp inconsistent).
    if cloudbench:
        autoconv = cloudbench_autoconversion(q_c)
    else:
        autoconv = config.autoconversion_rate * jnp.maximum(
            q_c - config.autoconversion_threshold, 0.0
        )

    # 3. Accretion: cloud collected by rain.  Fractional powers of q_r
    # have unbounded derivative at q_r=0 — safe_pow handles the AD guard.
    if cloudbench:
        accretion = cloudbench_accretion(q_c, q_r, rho)
    else:
        accretion = config.accretion_coeff * q_c * safe_pow(q_r, _KESSLER_ACCR_EXP)

    # 4. Evaporation of rain: Klemp & Wilhelmson (1978), capped by the
    # sub-saturation left after the cloud evaporates and by q_r/dt.
    # WRF's cap reads the scheme's HARD adjustment amount (``prod``), not the
    # sigmoid-smoothed condensation rate
    demand, _ = saturation_deficit_rate(T, q_v, p_full, _dt_safe)
    if cloudbench:
        evaporation = _cap_rain_evaporation(
            cloudbench_rain_evaporation_rate(T, q_v, q_r, q_sat, rho),
            q_r, q_c, demand, _dt_safe)
    else:
        evaporation = kw78_rain_evaporation(
            q_v, q_r, q_c, q_sat, p_full, rho, demand, _dt_safe,
            scale=config.evaporation_coeff,
        )

    # === Joint donor clamp on q_c sinks ===
    # Saturation-evaporation, autoconversion, and accretion can each
    # individually be donor-clamped, but the *combined* sink rate can
    # still exceed ``q_c / dt`` and drive ``q_c`` negative on an
    # explicit Euler step.  E.g. with ``q_c = 1e-4``, ``q_r = 5e-3``,
    # 80 % RH at 290 K, accretion + saturation-evap together over a
    # 1200-s step demanded ~26× the available q_c (Codex audit cycle
    # 2 / cycle 3 follow-up).  Mirror the Morrison/Thompson joint
    # donor clamp so total water is conserved when sinks compete.
    cond_evap_sink = jnp.maximum(-condensation, 0.0)
    qc_sink_total = cond_evap_sink + autoconv + accretion
    qc_avail = jnp.clip(q_c, 0.0)
    # Shared AD-safe donor clamp helper.  Replaces the previous
    # boolean ``sink_active`` double-where with an explicit
    # ``divisor_floor=1e-15`` that bounds the worst-case VJP under
    # fp32 to ``-q / 1e-30 ≈ -1e30`` (safely within fp32 dynamic
    # range).
    qc_scale = donor_clamp_scale(qc_avail, qc_sink_total, dt)
    autoconv = autoconv * qc_scale
    accretion = accretion * qc_scale
    # Scale the evaporation branch only; positive condensation
    # (saturation adjustment from supersaturation) is unaffected.
    condensation = jnp.where(
        condensation < 0.0, condensation * qc_scale, condensation,
    )
    dq_v_sat = -condensation
    dq_c_sat = condensation

    # 5. Rain sedimentation
    if cloudbench:
        V_t = chen2022_rain_fall_speed(q_r, rho)
    else:
        rho_sfc = rho[:, -1:]
        V_t = config.rain_fall_speed * jnp.sqrt(
            rho_sfc / jnp.clip(rho, _RHO_FLOOR)
        )
    # Joint q_r donor cap: pass evaporation as ``extra_sink`` so the
    # sedimentation flux limiter accounts for the rain evaporation that
    # also removes q_r in the same step.  Codex iter-29 #1.
    sed_tend, precipitation = sedimentation_tendency(
        q_r, rho, V_t, dz, dt=dt,
        return_surface_flux=True,
        extra_sink=evaporation,
    )

    # 6. Latent heating
    dT_dt = constants.L_v * (condensation - evaporation) / constants.c_pd

    # Combine tracer tendencies — sources match the scaled sinks so
    # total water (q_v + q_c + q_r) is conserved per layer (modulo
    # rain evaporation, which exchanges with q_v, and sedimentation,
    # which redistributes q_r vertically).
    dq_v_dt = dq_v_sat + evaporation
    dq_c_dt = dq_c_sat - autoconv - accretion
    dq_r_dt = autoconv + accretion - evaporation + sed_tend

    # Precipitation comes from the dt-limited surface flux returned by
    # ``sedimentation_tendency`` so column water conservation holds
    # exactly when the CFL limiter fires.  An earlier formulation
    # diagnosed precipitation as ``q_r_bot · ρ · V_t``, which exceeds
    # the actual amount removed from the column whenever
    # ``V_t·dt/dz_bot > 1``.

    # Pin dtype to the input precision so we never silently promote
    # the unused-tendency placeholders to f64 under x64 mode.
    _dtype = T.dtype
    z = jnp.zeros((ncol, nlev), dtype=_dtype)
    return MicrophysicsOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=dq_r_dt,
        dq_i_dt=z,
        dq_s_dt=z,
        dq_g_dt=z,
        dN_c_dt=z,
        dN_r_dt=z,
        dN_i_dt=z,
        precipitation=precipitation,
        precip_frozen=jnp.zeros_like(precipitation),   # L_v-only: all liquid
        # Positive saturation-adjustment condensation (the q_v sink that becomes
        # cloud water) for the coupled pipeline's joint vapour donor clamp.
        dq_v_to_qc_dt=jnp.maximum(dq_c_sat, 0.0),
    )
