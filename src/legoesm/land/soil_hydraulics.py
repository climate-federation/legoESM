"""Soil water retention curves and hydraulic conductivity.

Six pluggable retention curve models, all JAX-differentiable:
1. Clapp-Hornberger (1978) — faithful implementation
2. Van Genuchten-Mualem (1980) — faithful implementation
3. Brooks-Corey (1964) — faithful implementation
4. Campbell (1974) — faithful implementation
5. Peters-Durner-Iden (2015) — approximate variant: faithful SWRC and
   capillary conductivity, but uses VG inverse for psi(theta) and
   simplified film-flow conductivity (see pdi_psi, pdi_K docstrings)
6. Lu (2016) — approximate variant: faithful three-regime SWRC, but
   uses VG inverse for psi(theta) and simplified conductivity
   (see lu_psi, lu_K docstrings)

References
----------
- Clapp & Hornberger (1978): Empirical equations for some soil hydraulic properties.
- Van Genuchten (1980): A closed-form equation for predicting the hydraulic
  conductivity of unsaturated soils.
- Brooks & Corey (1964): Hydraulic properties of porous media.
- Campbell (1974): A simple method for determining unsaturated conductivity.
- Peters (2013): Simple consistent models for water retention and hydraulic
  conductivity in the complete moisture range. Water Resources Research, 49(10).
- Iden & Durner (2014): Comment on "Simple consistent models..." Water Resources
  Research, 50(9), 7530-7534.
- Iden, Peters & Durner (2015): Improving prediction of hydraulic conductivity by
  constraining capillary bundle models to a maximum pore size. Advances in Water
  Resources, 85, 86-92.
- Lu (2016): Generalized soil water retention equation for adsorption and
  capillarity. J. Geotech. Geoenviron. Eng., 142(10), 04016051.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

_TINY = float(jnp.finfo(jnp.float32).tiny)  # Smallest normal float32 (~1.18e-38)


class SoilHydraulicsConfig(NamedTuple):
    """Configuration for soil hydraulic properties."""
    retention_curve: str = "van_genuchten"
    # Van Genuchten defaults (loam)
    theta_r: float = 0.078
    theta_sat: float = 0.43
    alpha_vg: float = 3.6       # [1/m]
    n_vg: float = 1.56
    K_sat: float = 2.89e-6      # [m/s] (~0.25 m/day)
    # Clapp-Hornberger / Campbell defaults (loam)
    psi_sat: float = -0.478     # [m]
    b_ch: float = 5.39
    # Brooks-Corey defaults
    psi_b: float = -0.478       # air-entry pressure [m]
    lambda_bc: float = 0.186    # pore-size index
    # Peters-Durner-Iden (2015) defaults
    h0_pdi: float = 6.3e4       # suction at oven dryness [m] (10^4.8)
    h_crit: float = 0.06        # max pore suction [m] (Iden et al. 2015)
    tau_s: float = 0.1          # saturated tortuosity [-] (Mualem median ~0.095)
    omega_pdi: float = 0.5      # tortuosity-connectivity exponent [-]
    # Lu (2016) three-regime SWRC defaults (loam)
    theta_a: float = 0.02       # max adsorptive water content [m3/m3]
    h_a: float = 100.0          # adsorption strength [m] (suction scale)
    n_a: float = 0.5            # adsorption exponent [-]
    h_cav: float = 50.0         # mean cavitation suction [m]
    sigma_cav: float = 1.0      # cavitation spread (log-normal width) [-]
    # Elastic storage near saturation
    S_s: float = 1e-4           # specific storage [1/m]


# ==========================================================================
# Clapp-Hornberger (1978)
# ==========================================================================

def clapp_hornberger_psi(theta: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Matric potential from volumetric water content."""
    Se = jnp.clip(theta / config.theta_sat, 1e-6, 1.0)
    return config.psi_sat * Se ** (-config.b_ch)


def clapp_hornberger_theta(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Volumetric water content from matric potential."""
    # psi and psi_sat are both negative; ratio >= 1 for unsaturated soil
    ratio = jnp.clip(psi / config.psi_sat, 1.0, None)
    Se = ratio ** (-1.0 / config.b_ch)
    Se = jnp.clip(Se, 0.0, 1.0)
    return config.theta_sat * Se


def clapp_hornberger_K(theta: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Hydraulic conductivity."""
    Se = jnp.clip(theta / config.theta_sat, 1e-6, 1.0)
    return config.K_sat * Se ** (2.0 * config.b_ch + 3.0)


def clapp_hornberger_C(theta: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Specific moisture capacity dtheta/dpsi."""
    Se = jnp.clip(theta / config.theta_sat, 1e-6, 1.0)
    return -config.theta_sat / (config.b_ch * config.psi_sat) * Se ** (config.b_ch + 1.0)


# ==========================================================================
# Van Genuchten-Mualem (1980)
# ==========================================================================

def van_genuchten_Se(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Effective saturation from matric potential."""
    m = 1.0 - 1.0 / config.n_vg
    # psi < 0 unsaturated; psi >= 0 saturated
    abs_alpha_psi = jnp.abs(config.alpha_vg * psi)
    Se = (1.0 + abs_alpha_psi ** config.n_vg) ** (-m)
    # Saturated where psi >= 0
    Se = jnp.where(psi >= 0.0, 1.0, Se)
    return Se


def van_genuchten_theta(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Volumetric water content from matric potential."""
    Se = van_genuchten_Se(psi, config)
    return config.theta_r + (config.theta_sat - config.theta_r) * Se


def van_genuchten_psi(theta: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Matric potential from volumetric water content (inverse)."""
    Se = jnp.clip((theta - config.theta_r) / (config.theta_sat - config.theta_r),
                  1e-6, 1.0 - 1e-6)
    m = 1.0 - 1.0 / config.n_vg
    return -(1.0 / config.alpha_vg) * (Se ** (-1.0 / m) - 1.0) ** (1.0 / config.n_vg)


def van_genuchten_K(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Hydraulic conductivity from matric potential."""
    Se = van_genuchten_Se(psi, config)
    m = 1.0 - 1.0 / config.n_vg
    inner = 1.0 - (1.0 - Se ** (1.0 / m)) ** m
    return config.K_sat * jnp.sqrt(Se) * inner ** 2


def van_genuchten_C(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Specific moisture capacity dtheta/dpsi."""
    m = 1.0 - 1.0 / config.n_vg
    n = config.n_vg
    alpha = config.alpha_vg
    abs_alpha_psi = jnp.abs(alpha * psi)
    term = 1.0 + abs_alpha_psi ** n
    C = (alpha * m * n * abs_alpha_psi ** (n - 1.0)
         * (config.theta_sat - config.theta_r)
         * term ** (-m - 1.0))
    return jnp.where(psi >= 0.0, 0.0, C)


# ==========================================================================
# Brooks-Corey (1964) — smooth version
# ==========================================================================

def brooks_corey_Se(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Effective saturation (smooth at air entry)."""
    ratio = jnp.abs(config.psi_b) / jnp.clip(jnp.abs(psi), 1e-10, None)
    Se = jnp.clip(ratio ** config.lambda_bc, 0.0, 1.0)
    # Saturated where psi >= psi_b (less negative)
    Se = jnp.where(psi >= config.psi_b, 1.0, Se)
    return Se


def brooks_corey_theta(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Volumetric water content from matric potential."""
    Se = brooks_corey_Se(psi, config)
    return config.theta_r + (config.theta_sat - config.theta_r) * Se


def brooks_corey_psi(theta: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Matric potential from volumetric water content (inverse)."""
    Se = jnp.clip((theta - config.theta_r) / (config.theta_sat - config.theta_r),
                  1e-6, 1.0 - 1e-6)
    return config.psi_b * Se ** (-1.0 / config.lambda_bc)


def brooks_corey_K(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Hydraulic conductivity."""
    Se = brooks_corey_Se(psi, config)
    return config.K_sat * Se ** (3.0 + 2.0 / config.lambda_bc)


# ==========================================================================
# Campbell (1974)
# ==========================================================================

def campbell_psi(theta: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Matric potential from water content."""
    Se = jnp.clip(theta / config.theta_sat, 1e-6, 1.0)
    return config.psi_sat * Se ** (-config.b_ch)


def campbell_K(theta: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Hydraulic conductivity."""
    Se = jnp.clip(theta / config.theta_sat, 1e-6, 1.0)
    return config.K_sat * Se ** (2.0 * config.b_ch + 3.0)


# ==========================================================================
# Peters-Durner-Iden (2015) — capillary + adsorptive film flow
# ==========================================================================

def _pdi_Gamma(h: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Van Genuchten saturation function Gamma(h) used inside PDI."""
    m = 1.0 - 1.0 / config.n_vg
    alpha_h = config.alpha_vg * h
    return (1.0 + alpha_h ** config.n_vg) ** (-m)


def _pdi_ha(config: SoilHydraulicsConfig) -> float:
    """Suction where capillary saturation Sc = 0.75 (Iden & Durner 2014)."""
    m = 1.0 - 1.0 / config.n_vg
    Gamma_h0 = _pdi_Gamma(jnp.array(config.h0_pdi), config)
    # gamma = 0.75*(1 - Gamma(h0)) + Gamma(h0) = 0.75 + 0.25*Gamma(h0)
    gamma_val = 0.75 + 0.25 * Gamma_h0
    # ha = alpha^{-1} * (gamma^{-1/m} - 1)^{1/n}
    ha = (1.0 / config.alpha_vg) * (gamma_val ** (-1.0 / m) - 1.0) ** (1.0 / config.n_vg)
    return ha


def _pdi_Sc(h: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Rescaled capillary saturation Sc(h) (Peters 2013).

    Sc = [Gamma(h) - Gamma(h0)] / [1 - Gamma(h0)]
    """
    Gamma_h = _pdi_Gamma(h, config)
    Gamma_h0 = _pdi_Gamma(jnp.array(config.h0_pdi), config)
    Sc = (Gamma_h - Gamma_h0) / (1.0 - Gamma_h0 + _TINY)
    return jnp.clip(Sc, 0.0, 1.0)


def _pdi_Snc(h: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Non-capillary (adsorptive) saturation Snc(h) (Iden & Durner 2014).

    Snc = {ln(h0/h) - b*ln[1 + (h/ha)^(1/b)]} / ln(h0/ha)
    Transitions smoothly from 0 at h0 to 1 at ha.
    """
    h0 = config.h0_pdi
    ha = _pdi_ha(config)

    # Smoothing parameter b (Iden & Durner 2014)
    theta_range = config.theta_sat - config.theta_r + _TINY
    b1 = (config.theta_r / theta_range) ** 2
    b0 = 0.1 * jnp.log(10.0)
    b = b0 * (1.0 + 2.0 * (1.0 - jnp.exp(-b1 * config.n_vg ** 2)))

    h_safe = jnp.clip(h, 1e-10, h0)
    numerator = jnp.log(h0 / h_safe) - b * jnp.log(1.0 + (h_safe / ha) ** (1.0 / b))
    denominator = jnp.log(h0 / ha) + _TINY
    Snc = numerator / denominator
    # Clamp: Snc=1 for h <= ha, Snc=0 at h0
    Snc = jnp.clip(Snc, 0.0, 1.0)
    return Snc


def pdi_theta(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """PDI water content from matric potential.

    theta(h) = (theta_s - theta_r)*Sc(h) + theta_r*Snc(h)
    """
    h = jnp.abs(psi)  # suction head (positive)
    h = jnp.clip(h, 1e-10, config.h0_pdi)
    Sc = _pdi_Sc(h, config)
    Snc = _pdi_Snc(h, config)
    theta = (config.theta_sat - config.theta_r) * Sc + config.theta_r * Snc
    # Saturated when psi >= 0
    return jnp.where(psi >= 0.0, config.theta_sat, theta)


def pdi_psi(theta: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """PDI matric potential from water content (approximate VG inverse).

    NOTE: This uses the standard Van Genuchten inverse as an approximation.
    The capillary component dominates in the normal moisture range
    (theta_r < theta < theta_sat), so the VG inverse is a reasonable
    proxy. A faithful PDI inverse would require numerical root-finding
    (e.g., Newton iteration on the full pdi_theta function), which is
    not implemented here.
    """
    return van_genuchten_psi(theta, config)


def pdi_K(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """PDI hydraulic conductivity with max pore constraint (Iden et al. 2015).

    K(h) = Kc(h) + Knc(h)

    Capillary: VG-Mualem with Sc rescaling and h_crit constraint.
    Non-capillary: film flow proportional to (1 - Snc).
    """
    h = jnp.abs(psi)
    h = jnp.clip(h, 1e-10, config.h0_pdi)

    m = 1.0 - 1.0 / config.n_vg
    Sc = _pdi_Sc(h, config)

    # --- Capillary conductivity (VG-Mualem on Sc) ---
    # Kr,c = Sc^omega * [1 - (1 - Sc^(1/m))^m]^2
    Sc_safe = jnp.clip(Sc, 1e-10, 1.0 - 1e-10)
    inner = 1.0 - (1.0 - Sc_safe ** (1.0 / m)) ** m
    Kr_c = Sc_safe ** config.omega_pdi * inner ** 2

    # Max pore constraint (Iden et al. 2015): for h < h_crit,
    # interpolate K smoothly to K_sat at h=0
    h_crit = config.h_crit
    Sc_crit = _pdi_Sc(jnp.array(h_crit), config)
    Sc_crit_safe = jnp.clip(Sc_crit, 1e-10, 1.0 - 1e-10)
    inner_crit = 1.0 - (1.0 - Sc_crit_safe ** (1.0 / m)) ** m
    Kr_crit = Sc_crit_safe ** config.omega_pdi * inner_crit ** 2

    # Cosine interpolation between h_crit and h=0
    x = jnp.log10(jnp.clip(h, 1e-10, None))
    x_crit = jnp.log10(h_crit)
    x_s = jnp.log10(1e-10)  # effectively h ~ 0
    frac = jnp.clip((x - x_s) / (x_crit - x_s + _TINY), 0.0, 1.0)
    Kr_interp = 1.0 + 0.5 * (1.0 + jnp.cos(jnp.pi * frac)) * (Kr_crit - 1.0)

    # Use interpolated K when h < h_crit, otherwise standard capillary
    Kc = config.K_sat * jnp.where(h < h_crit, Kr_interp, Kr_c)

    # --- Non-capillary (film) conductivity ---
    # Knc = c_film * theta_r * (1 - Snc) * scale
    # Simplified: proportional to K_sat * (1 - Snc) * (theta_r/theta_sat)
    Snc = _pdi_Snc(h, config)
    c_film = 1.35e-8  # m^(5/2)/s (Peters 2013)
    ha = _pdi_ha(config)
    film_scale = ha ** (-1.5) - config.h0_pdi ** (-1.5)
    Knc = c_film * config.theta_r * jnp.clip(film_scale, 0.0, None) * (1.0 - Snc)

    K_total = Kc + Knc
    return jnp.where(psi >= 0.0, config.K_sat, K_total)


def pdi_C(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """PDI specific moisture capacity dtheta/dpsi (finite difference)."""
    eps = 1e-4
    theta_p = pdi_theta(psi + eps, config)
    theta_m = pdi_theta(psi - eps, config)
    return (theta_p - theta_m) / (2.0 * eps)


# ==========================================================================
# Lu (2016) — generalized three-regime SWRC
# ==========================================================================

def _lu_adsorption(h: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Adsorptive water content component (Lu 2016).

    theta_a(h) = theta_a_max * exp(-(h / h_a)^n_a)

    Represents tightly bound adsorbed film on particle surfaces.
    """
    return config.theta_a * jnp.exp(-(h / config.h_a) ** config.n_a)


def _lu_cavitation(h: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Cavitation probability function P_cav(h) (Lu 2016).

    P_cav = 0.5 * erfc[ln(h/h_cav) / (sigma_cav * sqrt(2))]

    Describes the probability that a capillary meniscus still exists at suction h.
    """
    ln_ratio = jnp.log(jnp.clip(h, 1e-10, None) / config.h_cav)
    from jax.scipy.special import erfc
    return 0.5 * erfc(ln_ratio / (config.sigma_cav * jnp.sqrt(2.0)))


def lu_theta(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Lu (2016) three-regime water content.

    theta(h) = theta_a(h) + [theta_cap(h) - theta_a(h)] * P_cav(h)

    where theta_cap is the standard VG capillary retention and P_cav
    modulates the transition from capillary to adsorptive regime.
    """
    h = jnp.abs(psi)
    h = jnp.clip(h, 1e-10, None)

    # Capillary retention (standard VG)
    theta_cap = van_genuchten_theta(-h, config)

    # Adsorptive retention
    theta_a = _lu_adsorption(h, config)

    # Cavitation probability
    P_cav = _lu_cavitation(h, config)

    theta = theta_a + (theta_cap - theta_a) * P_cav
    return jnp.where(psi >= 0.0, config.theta_sat, theta)


def lu_psi(theta: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Lu (2016) inverse: matric potential from water content (approximate).

    NOTE: Uses the standard Van Genuchten inverse as an approximation.
    The capillary regime dominates in the normal moisture range relevant
    for the Richards equation. A faithful Lu inverse would require
    numerical root-finding on the full lu_theta function (capillary +
    adsorptive + cavitation), which is not implemented here.
    """
    return van_genuchten_psi(theta, config)


def lu_K(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Lu (2016) hydraulic conductivity (approximate).

    NOTE: Uses VG-Mualem for the capillary component with a film-flow
    floor in the adsorptive regime. The original Lu (2016) formulation
    includes a more elaborate conductivity model coupling the capillary,
    adsorptive, and cavitation regimes. This is a simplified variant
    adequate for Richards equation integration in the normal moisture
    range.
    """
    theta = lu_theta(psi, config)
    Se = jnp.clip(
        (theta - config.theta_r) / (config.theta_sat - config.theta_r + _TINY),
        1e-10, 1.0,
    )
    m = 1.0 - 1.0 / config.n_vg
    inner = 1.0 - (1.0 - Se ** (1.0 / m)) ** m
    K_cap = config.K_sat * jnp.sqrt(Se) * inner ** 2
    # Film-flow floor: prevents K=0 in the adsorptive regime
    K_film = config.K_sat * 1e-10
    K = jnp.maximum(K_cap, K_film)
    return jnp.where(psi >= 0.0, config.K_sat, K)


def lu_C(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Lu (2016) specific moisture capacity (finite difference)."""
    eps = 1e-4
    theta_p = lu_theta(psi + eps, config)
    theta_m = lu_theta(psi - eps, config)
    return (theta_p - theta_m) / (2.0 * eps)


# ==========================================================================
# Dispatch functions
# ==========================================================================

def theta_from_psi(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Compute theta from psi using the configured retention curve."""
    curve = config.retention_curve
    if curve == "van_genuchten":
        return van_genuchten_theta(psi, config)
    elif curve == "clapp_hornberger" or curve == "campbell":
        return clapp_hornberger_theta(psi, config)
    elif curve == "brooks_corey":
        return brooks_corey_theta(psi, config)
    elif curve == "pdi":
        return pdi_theta(psi, config)
    elif curve == "lu":
        return lu_theta(psi, config)
    else:
        raise ValueError(f"Unknown retention curve: {curve}")


def psi_from_theta(theta: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Compute psi from theta using the configured retention curve."""
    curve = config.retention_curve
    if curve == "van_genuchten":
        return van_genuchten_psi(theta, config)
    elif curve == "clapp_hornberger" or curve == "campbell":
        return clapp_hornberger_psi(theta, config)
    elif curve == "brooks_corey":
        return brooks_corey_psi(theta, config)
    elif curve == "pdi":
        return pdi_psi(theta, config)
    elif curve == "lu":
        return lu_psi(theta, config)
    else:
        raise ValueError(f"Unknown retention curve: {curve}")


def hydraulic_conductivity(psi: jnp.ndarray, theta: jnp.ndarray,
                           config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Compute hydraulic conductivity using the configured retention curve."""
    curve = config.retention_curve
    if curve == "van_genuchten":
        return van_genuchten_K(psi, config)
    elif curve == "clapp_hornberger" or curve == "campbell":
        return clapp_hornberger_K(theta, config)
    elif curve == "brooks_corey":
        return brooks_corey_K(psi, config)
    elif curve == "pdi":
        return pdi_K(psi, config)
    elif curve == "lu":
        return lu_K(psi, config)
    else:
        raise ValueError(f"Unknown retention curve: {curve}")


def moisture_capacity(psi: jnp.ndarray, theta: jnp.ndarray,
                      config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Compute specific moisture capacity dtheta/dpsi.

    Includes elastic storage term near saturation for regularization.
    """
    curve = config.retention_curve
    if curve == "van_genuchten":
        C = van_genuchten_C(psi, config)
    elif curve == "clapp_hornberger" or curve == "campbell":
        C = clapp_hornberger_C(theta, config)
    elif curve == "pdi":
        C = pdi_C(psi, config)
    elif curve == "lu":
        C = lu_C(psi, config)
    else:
        # Brooks-Corey: use finite difference approximation
        eps = 1e-4
        theta_p = theta_from_psi(psi + eps, config)
        theta_m = theta_from_psi(psi - eps, config)
        C = (theta_p - theta_m) / (2.0 * eps)

    # Add elastic storage near saturation
    C = C + config.S_s * config.theta_sat
    return C


def interblock_K(K_above: jnp.ndarray, K_below: jnp.ndarray) -> jnp.ndarray:
    """Geometric mean of hydraulic conductivity between adjacent layers."""
    return jnp.sqrt(jnp.clip(K_above, 1e-20, None) * jnp.clip(K_below, 1e-20, None))
