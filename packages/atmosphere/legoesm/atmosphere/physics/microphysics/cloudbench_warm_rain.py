"""CloudBench one-moment warm-rain microphysics column backend
(``MicrophysicsConfig.scheme="cloudbench_warm_rain"``).

The CloudBench LES one-moment warm rain as coded in swirl-lm: threshold
autoconversion, CloudMicrophysics-1M accretion and rain evaporation with the
Azimi et al. (2024) EKI coefficients, and the Chen et al. (2022) rain fall speed
(with swirl-lm's radius-for-diameter substitution reproduced on purpose).
Formulas, values and every source line are in the constants block below.

A scheme of its own, beside the faithful ESCOMP Kessler port in ``kessler.py``
(user decision 2026-11-19); it shares only the ``_warm_rain`` saturation
adjustment and donor clamp and the ``output.sedimentation_tendency`` operator.
The saturation adjustment is the smooth (sigmoid) form for differentiability;
the scheme is NOT globally smooth -- it keeps max/min/clip kinks (donor clamps,
positivity floors).  Pins: ``tests/atmosphere/hydrostatic/unit/
test_cloudbench_warm_rain.py``.
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
from legoesm.atmosphere.physics.microphysics.config import CloudBenchWarmRainConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    sedimentation_tendency,
)


_RHO_FLOOR = 0.1                 # density floor aloft [kg/m^3]
_QSAT_FLOOR = 1.0e-10            # division guard on q_sat [kg/kg]

# =============================================================================
# CloudBench one-moment warm-rain constants
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

__physics_contract__ = {
    "summary": (
        "CloudBench one-moment warm-rain microphysics (swirl-lm forms): "
        "saturation adjustment (cloud condensation/evaporation), threshold "
        "cloud->rain autoconversion, accretion, rain evaporation and rain "
        "sedimentation to surface precipitation."
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
    "reference": (
        "Chammas et al. (2026), Sci. Adv., doi 10.1126/sciadv.aec8488; Azimi et "
        "al. (2024); Chen et al. (2022)"
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_cloudbench_warm_rain.py -- "
        "process rates against the published forms, column water + surface rain "
        "and moist enthalpy closure, dispatch, differentiable wrt T, q_v."
    ),
}


def _cap_rain_evaporation(rate, q_r, q_c, demand, dt):
    """Rain may evaporate only the sub-saturation left after all cloud water has
    evaporated, ``max(-demand - q_c/dt, 0)`` (WRF ``max(-prod - qc, 0)``), and at
    most ``q_r/dt``. Scales a rate only, so water and enthalpy still close."""
    dt_s = jnp.maximum(dt, 1.0e-12)
    rate = jnp.minimum(rate, jnp.clip(-demand - jnp.clip(q_c, 0.0) / dt_s, 0.0))
    return jnp.minimum(rate, jnp.clip(q_r, 0.0, None) / dt_s)


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
    lv = constants.L_v  # latent-ok: CloudBench warm-rain oracle constant
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


def cloudbench_warm_rain_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: CloudBenchWarmRainConfig = CloudBenchWarmRainConfig(),
) -> MicrophysicsOutput:
    """CloudBench warm-rain tendencies; arrays (ncol, nlev), ``p_half``
    (ncol, nlev+1), surface at ``[:, -1]``.  Only ``q_c`` and ``q_r`` of
    ``hydrometeors`` are used."""
    ncol, nlev = T.shape
    q_c = hydrometeors.q_c
    q_r = hydrometeors.q_r

    # 1. Saturation adjustment (shared helper), with condensation donor-clamped
    # by the available q_v.
    _dt_safe = jnp.maximum(dt, 1e-10)
    condensation, q_sat = saturation_adjustment(
        T, q_v, p_full, _dt_safe, config.saturation_sharpness, q_c=q_c,
        hard_adjust=config.hard_saturation_adjustment,
        hard_threshold=config.hard_sat_adjust_threshold,
        hard_max_heating_K=config.hard_sat_max_heating_K,
        l_v=constants.L_v,  # latent-ok: CloudBench warm-rain oracle constant
    )
    condensation = jnp.minimum(condensation, jnp.clip(q_v, 0.0, None) / _dt_safe)

    # 2-3. Autoconversion and accretion, from raw q_c so the joint clamp below
    # scales every q_c sink consistently.
    autoconv = cloudbench_autoconversion(q_c)
    accretion = cloudbench_accretion(q_c, q_r, rho)

    # 4. Rain evaporation, capped by the UNSMOOTHED sub-saturation left after
    # the cloud evaporates and by q_r/dt.
    demand, _ = saturation_deficit_rate(T, q_v, p_full, _dt_safe, l_v=constants.L_v)  # latent-ok: CloudBench warm-rain oracle constant
    evaporation = _cap_rain_evaporation(
        cloudbench_rain_evaporation_rate(T, q_v, q_r, q_sat, rho),
        q_r, q_c, demand, _dt_safe)

    # Joint donor clamp on the q_c sinks (evaporation branch of the adjustment,
    # autoconversion, accretion): scales rates only, so water still closes.
    qc_scale = donor_clamp_scale(
        jnp.clip(q_c, 0.0), jnp.maximum(-condensation, 0.0) + autoconv + accretion, dt)
    autoconv = autoconv * qc_scale
    accretion = accretion * qc_scale
    condensation = jnp.where(condensation < 0.0, condensation * qc_scale, condensation)

    # 5. Rain sedimentation; evaporation passed as the joint q_r sink so the
    # dt-limited flux and the returned surface flux conserve column water.
    V_t = chen2022_rain_fall_speed(q_r, rho)
    sed_tend, precipitation = sedimentation_tendency(
        q_r, rho, V_t, dz, dt=dt,
        return_surface_flux=True,
        extra_sink=evaporation,
    )

    # 6. Latent heating and tracer tendencies (sources match the scaled sinks).
    dT_dt = constants.L_v * (condensation - evaporation) / constants.c_pd  # latent-ok: CloudBench warm-rain oracle constant
    dq_v_dt = -condensation + evaporation
    dq_c_dt = condensation - autoconv - accretion
    dq_r_dt = autoconv + accretion - evaporation + sed_tend

    z = jnp.zeros((ncol, nlev), dtype=T.dtype)
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
        dq_v_to_qc_dt=jnp.maximum(condensation, 0.0),
    )
