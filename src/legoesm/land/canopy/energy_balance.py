"""Leaf and soil energy balance for the two-leaf canopy model.

Two schemes are provided:
  - BT (Bulk Transfer): direct specific-humidity-gradient formulation.
      LE = λ ρ (q_f - q_c) / (Rb + rs)
    Computationally efficient; default.
  - PM (Penman-Monteith): second-order Paw & Gao (1988) solution.
    More physical but requires second derivatives of sat. vapour pressure.

Canopy air-space update: conductance-weighted mixing of heat/vapour from
leaves, soil, and free atmosphere.

All functions are pure JAX, JIT-compatible, and differentiable.
Sources: DifferBESS/process/CanopyEnergyBalance.py, Soil.py, CarbonWaterFluxes.py
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

# Physical constants
_T0    = 273.15      # [K]
_Ps0   = 101325.0    # standard pressure [Pa]
_Lv    = 2.501e6     # latent heat of vaporisation at 0°C [J kg-1]
_sigma = 5.670373e-8  # Stefan-Boltzmann [W m-2 K-4]


# ---------------------------------------------------------------------------
# Meteorological helpers
# ---------------------------------------------------------------------------

@jax.jit
def saturation_specific_humidity(T: jax.Array, p: jax.Array) -> jax.Array:
    """Saturation specific humidity [kg kg-1] from T [K] and p [Pa]."""
    e_s = 611.2 * jnp.exp(17.67 * (T - _T0) / ((T - _T0) + 243.5))
    return 0.622 * e_s / (p - (1.0 - 0.622) * e_s)


@jax.jit
def canopy_met_variables(
    Ps: jax.Array,
    Tc: jax.Array,
    q_c: jax.Array,
) -> tuple[jax.Array, ...]:
    """Meteorological variables for the canopy air space.

    Parameters
    ----------
    Ps  : atmospheric pressure [Pa]
    Tc  : canopy air temperature [K]
    q_c : canopy air specific humidity [kg kg-1]

    Returns
    -------
    e_c, es_c, VPD_c, RH_c, desTc, ddesTc, gamma
      All in Pa (or Pa K-1 for derivatives; Pa K-2 for second derivative).
    """
    # Vapour pressure from specific humidity
    e_c  = q_c * Ps / (0.622 + (1.0 - 0.622) * q_c)
    # Saturation vapour pressure (Clausius-Clapeyron)
    TcC  = Tc - _T0
    es_c = 611.2 * jnp.exp(17.67 * TcC / (TcC + 243.5))

    VPD_c = es_c - e_c
    RH_c  = jnp.clip(e_c / jnp.maximum(es_c, 1e-6), 0.0, 1.0)

    # First derivative des/dT [Pa K-1]
    desTc  = es_c * 4098.0 * (TcC + 237.3) ** (-2)
    # Second derivative d²es/dT² [Pa K-2]
    ddesTc = 4098.0 * (
        desTc  * (TcC + 237.3) ** (-2)
        + (-2.0) * e_c * (TcC + 237.3) ** (-3)
    )

    # Latent heat and psychrometric constant
    lam   = _Lv - 2.361e3 * TcC
    gamma = 1004.0 / 0.622 * Ps / lam   # [Pa K-1]

    return e_c, es_c, VPD_c, RH_c, desTc, ddesTc, gamma


# ---------------------------------------------------------------------------
# Stomatal conductance (Ball-Berry)
# ---------------------------------------------------------------------------

def _compute_gs_and_ci(
    An: jax.Array,
    RH_c: jax.Array,
    Ca: jax.Array,
    Tf: jax.Array,
    Ps: jax.Array,
    m: jax.Array,
    b0: jax.Array,
    is_c4: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Ball-Berry stomatal conductance and intercellular CO2.

    Returns (rs [s m-1], gs [m s-1], Ci [μmol mol-1])
    """
    # Unit conversion: mol m-2 s-1 → m s-1
    cf = 0.446 * (_T0 / Tf) * (Ps / _Ps0)

    gs_mol = jnp.maximum(m * RH_c * An / jnp.maximum(Ca, 1e-3) + b0, b0)
    Ci = Ca - 1.6 * An / jnp.maximum(gs_mol, 1e-9)

    # Clip Ci to physically reasonable range
    if is_c4:
        Ci = jnp.clip(Ci, 0.2 * Ca, 0.6 * Ca)
    else:
        Ci = jnp.clip(Ci, 0.5 * Ca, 0.9 * Ca)

    rs = 1.0 / (gs_mol / cf * 1e-2)   # [s m-1]
    gs = 1.0 / rs                      # [m s-1]
    return rs, gs, Ci


# ---------------------------------------------------------------------------
# Leaf energy balance — BT (Bulk Transfer)
# ---------------------------------------------------------------------------

@jax.jit
def leaf_energy_balance_bt(
    An: jax.Array,
    ASW: jax.Array,
    ALW: jax.Array,
    Tf: jax.Array,
    Ps: jax.Array,
    Ca: jax.Array,
    Tc: jax.Array,
    q_f: jax.Array,
    q_c: jax.Array,
    RH_c: jax.Array,
    lam: jax.Array,
    Cp: jax.Array,
    rhoa: jax.Array,
    Rb: jax.Array,
    m: jax.Array,
    b0: jax.Array,
) -> tuple[jax.Array, ...]:
    """Leaf energy balance via direct bulk transfer (BT).

    LE = λ ρ (q_f - q_c) / (Rb + rs)
    H  = Rn - LE
    Tf_new = Tc + Rb / (ρ Cp) * H

    Parameters
    ----------
    An   : net photosynthesis [μmol m-2 s-1]
    ASW  : absorbed shortwave [W m-2]
    ALW  : net absorbed longwave [W m-2]
    Tf   : leaf temperature [K]
    Ps   : pressure [Pa]
    Ca   : ambient CO2 [μmol mol-1]
    Tc   : canopy air temperature [K]
    q_f  : leaf saturation specific humidity [kg kg-1]
    q_c  : canopy air specific humidity [kg kg-1]
    RH_c : canopy relative humidity [-]
    lam  : latent heat of vaporisation [J kg-1]
    Cp   : specific heat of air [J kg-1 K-1]
    rhoa : air density [kg m-3]
    Rb   : boundary-layer resistance [s m-1]
    m, b0: Ball-Berry slope and intercept [mol m-2 s-1 units]

    Returns
    -------
    Rn, LE, H, Tf_new, gs, Ci
    """
    rs, gs, Ci = _compute_gs_and_ci(An, RH_c, Ca, Tf, Ps, m, b0)

    Rn = ASW + ALW
    LE = lam * rhoa * (q_f - q_c) / jnp.maximum(Rb + rs, 1e-6)

    # Constrain: LE in [0, Rn] for daytime, 0 if below freezing
    LE = jnp.clip(LE, 0.0, jnp.where(Rn > 0.0, Rn, 0.0))
    LE = jnp.where(Tc < _T0, 0.0, LE)

    H  = Rn - LE
    dT = jnp.clip(Rb / (rhoa * Cp) * H, -30.0, 30.0)
    Tf_new = Tc + dT

    return Rn, LE, H, Tf_new, gs, Ci


# ---------------------------------------------------------------------------
# Leaf energy balance — PM (Penman-Monteith, second-order Paw & Gao 1988)
# ---------------------------------------------------------------------------

@jax.jit
def leaf_energy_balance_pm(
    An: jax.Array,
    ASW: jax.Array,
    ALW: jax.Array,
    Tf: jax.Array,
    Ps: jax.Array,
    Ca: jax.Array,
    Tc: jax.Array,
    VPD_c: jax.Array,
    RH_c: jax.Array,
    desTc: jax.Array,
    ddesTc: jax.Array,
    gamma_c: jax.Array,
    Cp: jax.Array,
    rhoa: jax.Array,
    Rb: jax.Array,
    m: jax.Array,
    b0: jax.Array,
) -> tuple[jax.Array, ...]:
    """Leaf energy balance via second-order Penman-Monteith (Paw & Gao 1988).

    Returns
    -------
    Rn, LE, H, Tf_new, gs, Ci
    """
    rs, gs, Ci = _compute_gs_and_ci(An, RH_c, Ca, Tf, Ps, m, b0)

    Rn = ASW + ALW
    rc = rs

    ddesTc_Rb2          = ddesTc * Rb**2
    gamma_Rb_rc          = gamma_c * (Rb + rc)
    rhoa_Cp_gamma_Rb_rc  = rhoa * Cp * gamma_Rb_rc

    a = 0.5 * ddesTc_Rb2 / rhoa_Cp_gamma_Rb_rc
    b = (-1.0
         - Rb * desTc / gamma_Rb_rc
         - ddesTc_Rb2 * Rn / rhoa_Cp_gamma_Rb_rc)
    c = (rhoa * Cp / gamma_Rb_rc * VPD_c
         + desTc * Rb / gamma_Rb_rc * Rn
         + 0.5 * ddesTc_Rb2 / rhoa_Cp_gamma_Rb_rc * Rn**2)

    disc = jnp.maximum(b**2 - 4.0 * a * c, 0.0)
    LE   = (-b + jnp.sign(b) * jnp.sqrt(disc)) / (2.0 * a)

    LE = jnp.clip(LE, 0.0, jnp.where(Rn > 0.0, Rn, 0.0))
    LE = jnp.where(Tc < _T0, 0.0, LE)

    H  = Rn - LE
    dT = jnp.clip(Rb / (rhoa * Cp) * H, -30.0, 30.0)
    Tf_new = Tc + dT

    return Rn, LE, H, Tf_new, gs, Ci


# ---------------------------------------------------------------------------
# Soil energy balance — BT
# ---------------------------------------------------------------------------

@jax.jit
def soil_energy_balance_bt(
    Ts: jax.Array,
    Tc: jax.Array,
    q_s: jax.Array,
    q_c: jax.Array,
    lam: jax.Array,
    rhoa: jax.Array,
    Cp: jax.Array,
    rah_soil: jax.Array,
    raw_soil: jax.Array,
    Rsoil: jax.Array,
    ASW_soil: jax.Array,
    ALW_soil: jax.Array,
    G_alpha: float | jax.Array,
) -> tuple[jax.Array, ...]:
    """Soil energy balance via bulk transfer.

    Rsoil = raw_below * (1/fStress_soil - 1) adds resistance proportional
    to soil dryness, reducing soil evaporation.

    Returns
    -------
    Rn_soil, LE_soil, H_soil, Ts_new, G
    """
    Rn  = ASW_soil + ALW_soil
    G   = G_alpha * Rn
    AE  = Rn - G

    LE  = lam * rhoa * (q_s - q_c) / jnp.maximum(raw_soil + Rsoil, 1e-6)

    LE  = jnp.clip(LE, 0.0, jnp.where(AE > 0.0, AE, 0.0))
    LE  = jnp.where(Tc < _T0, 0.0, LE)

    H   = AE - LE
    dT  = jnp.clip(rah_soil / (rhoa * Cp) * H, -30.0, 30.0)
    Ts_new = Tc + dT

    return Rn, LE, H, Ts_new, G


# ---------------------------------------------------------------------------
# Soil energy balance — PM
# ---------------------------------------------------------------------------

@jax.jit
def soil_energy_balance_pm(
    Ts: jax.Array,
    Tc: jax.Array,
    VPD_c: jax.Array,
    desTc: jax.Array,
    ddesTc: jax.Array,
    gamma_c: jax.Array,
    rhoa: jax.Array,
    Cp: jax.Array,
    rah_soil: jax.Array,
    raw_soil: jax.Array,
    Rsoil: jax.Array,
    ASW_soil: jax.Array,
    ALW_soil: jax.Array,
    G_alpha: float | jax.Array,
) -> tuple[jax.Array, ...]:
    """Soil energy balance via second-order Penman-Monteith.

    Returns
    -------
    Rn_soil, LE_soil, H_soil, Ts_new, G
    """
    Rn  = ASW_soil + ALW_soil
    G   = G_alpha * Rn
    AE  = Rn - G

    ddesTc_Raw2          = ddesTc * raw_soil**2
    gamma_Raw_Rsoil       = gamma_c * (raw_soil + Rsoil)
    rhoa_Cp_gamma         = rhoa * Cp * gamma_Raw_Rsoil

    a = 0.5 * ddesTc_Raw2 / rhoa_Cp_gamma
    b = (-1.0
         - rah_soil * desTc / gamma_Raw_Rsoil
         - ddesTc_Raw2 * AE / rhoa_Cp_gamma)
    c = (rhoa * Cp / gamma_Raw_Rsoil * VPD_c
         + desTc * rah_soil / gamma_Raw_Rsoil * AE
         + 0.5 * ddesTc_Raw2 / rhoa_Cp_gamma * AE**2)

    disc = jnp.maximum(b**2 - 4.0 * a * c, 0.0)
    LE   = (-b + jnp.sign(b) * jnp.sqrt(disc)) / (2.0 * a)

    LE  = jnp.clip(LE, 0.0, jnp.where(AE > 0.0, AE, 0.0))
    LE  = jnp.where(Tc < _T0, 0.0, LE)

    H   = AE - LE
    dT  = jnp.clip(rah_soil / (rhoa * Cp) * H, -30.0, 30.0)
    Ts_new = Tc + dT

    return Rn, LE, H, Ts_new, G


# ---------------------------------------------------------------------------
# Canopy air temperature and humidity update
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("coupling_scheme",))
def canopy_air_update(
    Ta: jax.Array,
    q_atm: jax.Array,
    Tf_Sun: jax.Array,
    Tf_Sh: jax.Array,
    Ts: jax.Array,
    gs_Sun: jax.Array,
    gs_Sh: jax.Array,
    Rb_Sun: jax.Array,
    Rb_Sh: jax.Array,
    rah_above: jax.Array,
    raw_above: jax.Array,
    rah_below: jax.Array,
    raw_below: jax.Array,
    Rsoil: jax.Array,
    Ps: jax.Array,
    coupling_scheme: str,
) -> tuple[jax.Array, jax.Array]:
    """Update canopy air temperature Tc and specific humidity q_c.

    Uses conductance-weighted mixing (DifferBESS CarbonWaterFluxes.py).

    Parameters
    ----------
    coupling_scheme : "FULLY_COUPLED" | "VEG_ONLY" | "LEAVES_ATMO"
      (static Python string — not traced)

    Returns
    -------
    Tc_new, q_c_new
    """
    ch_a   = 1.0 / jnp.maximum(rah_above, 1e-9)
    ch_sun = 1.0 / jnp.maximum(Rb_Sun,    1e-9)
    ch_sh  = 1.0 / jnp.maximum(Rb_Sh,     1e-9)

    gs_Sun_safe = jnp.maximum(gs_Sun, 1e-9)
    gs_Sh_safe  = jnp.maximum(gs_Sh,  1e-9)
    cw_a   = 1.0 / jnp.maximum(raw_above, 1e-9)
    cw_sun = 1.0 / (Rb_Sun + 1.0 / gs_Sun_safe)
    cw_sh  = 1.0 / (Rb_Sh  + 1.0 / gs_Sh_safe)

    q_f_Sun = saturation_specific_humidity(Tf_Sun, Ps)
    q_f_Sh  = saturation_specific_humidity(Tf_Sh,  Ps)
    q_s     = saturation_specific_humidity(Ts,     Ps)

    if coupling_scheme == "VEG_ONLY":
        # Canopy air excludes soil contribution
        Tc_new = (ch_a * Ta + ch_sun * Tf_Sun + ch_sh * Tf_Sh) / (
            ch_a + ch_sun + ch_sh)
        q_c_new = (cw_a * q_atm + cw_sun * q_f_Sun + cw_sh * q_f_Sh) / (
            cw_a + cw_sun + cw_sh)
    else:
        # FULLY_COUPLED or LEAVES_ATMO: soil included
        ch_g  = 1.0 / jnp.maximum(rah_below, 1e-9)
        cw_g  = 1.0 / jnp.maximum(raw_below + Rsoil, 1e-9)
        Tc_new = (ch_a * Ta + ch_sun * Tf_Sun + ch_sh * Tf_Sh + ch_g * Ts) / (
            ch_a + ch_sun + ch_sh + ch_g)
        q_c_new = (cw_a * q_atm + cw_sun * q_f_Sun + cw_sh * q_f_Sh + cw_g * q_s) / (
            cw_a + cw_sun + cw_sh + cw_g)

    return Tc_new, q_c_new
