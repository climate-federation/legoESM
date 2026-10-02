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

Faithfulness
------------
The Clapp-Hornberger, van Genuchten-Mualem and Brooks-Corey retention/
conductivity closed forms are pinned by
``tests/land/unit/test_soil_hydraulics_faithful.py`` against an independent
NumPy transcription of their canonical papers (relative tol 1e-9):

  * Clapp & Hornberger (1978) CAMPBELL power-law branch:
    psi = psi_sat*(theta/theta_sat)^(-b), K = K_sat*(theta/theta_sat)^(2b+3),
    C = dtheta/dpsi.  (CH also specify a short parabolic near-saturation section;
    legoESM adopts the pure Campbell power law everywhere, as CLM/Noah do — that
    is the branch pinned.)  Cross-checked against the on-disk gSAM Simple-Land-
    Model reference code ``SLM/soil_proc.f90``: for an unfrozen interface with
    UNIFORM soilw/Se (gSAM applies the node-k Bconst(k) to BOTH the soilw(k) and
    soilw(k+1) terms, so its depth-weighted two-node average reduces to soilw^p)
    its soil-water DIFFUSIVITY ``ks*B*|psi_sat|*soilw^(B+2)/poro`` equals this
    K*|dpsi/dtheta| = K/|C|, and its pore VELOCITY ``ks*soilw^(2B+2)/poro``
    equals K/(poro*Se) — so the exponents (2b+3 for K, b+2 for D) are fixed by
    both the paper and the on-disk model.
  * van Genuchten (1980) eqs 8-9: Se = (1 + |alpha psi|^n)^(-m), m = 1 - 1/n,
    K = K_sat*sqrt(Se)*[1 - (1 - Se^(1/m))^m]^2 (Mualem).
  * Brooks & Corey (1964): Se = (|psi_b|/|psi|)^lambda,
    K = K_sat*Se^(3 + 2/lambda) (Brooks-Corey/Burdine, eta = (2+3 lambda)/lambda).
  * ``retention_curve="campbell"`` dispatches to the Clapp-Hornberger power law
    (Campbell 1974 is that same psi_sat*Se^(-b) / K_sat*Se^(2b+3) form).

DEPARTURES / NUMERICS (NOT the pure closed form; documented, some canaried):
  * effective-saturation clips plateau K/psi at extreme dryness instead of the
    power law's 0/-inf limit.  The bounds are model-specific: CH floors Se at
    1e-6 (canaried); the vG/BC INVERSES clip Se to [1e-6, 1-1e-6]; vG K uses a
    sub-physical 1e-12 Se floor; Lu K clips Se to 1e-10 (plus a K_sat*1e-10 film
    floor); PDI applies its own bounds (see each fn).
  * the van-Genuchten/Lu K use a where-before-pow AD guard so the forward is
    EXACT at saturation (Se=1 -> K=K_sat) while the pow's infinite-slope branch
    is masked (finite reverse-mode gradient); canaried via K(psi=0)==K_sat.
  * the S_s elastic-storage branch in ``theta_from_psi``/``moisture_capacity``
    (theta = theta_sat + S_s*theta_sat*psi for psi>=0) is a ParFlow/CliMA
    change-of-variable near saturation, NOT a retention-curve term.
  * PDI (Peters-Durner-Iden 2015) and Lu (2016) are APPROXIMATE: faithful SWRC
    but a VG inverse for psi(theta) and a simplified film conductivity (see the
    ``pdi_psi``/``pdi_K``/``lu_psi``/``lu_K`` docstrings) — not pinned here.

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

# Fixed hydraulic-fit coefficients (not tunable).
_GAMMA_VAL_OFFSET = 0.75   # macroscopic capillary gamma offset
_B0_LOG10_COEFF = 0.1      # Campbell b-exponent log10 slope

_TINY = float(jnp.finfo(jnp.float32).tiny)  # Floor for denominators (~1.18e-38).
# This is used as a static guard against division by zero.  It is safe
# for both float32 and float64 because it is only ever compared to
# absolute values — float64 tiny is *smaller*, so the float32 constant
# is a conservative (larger) floor that works in both precisions.


__param_spec__ = {
    "SoilHydraulicsConfig": {
        "scheme_key": "land.soil_hydraulics",
        "excluded": {
            "S_s": "numerics: specific storage regulariser",
            "k_sat_decay_m": "opt-in depth-decay length (Niu 2005); 0 disables — "
                             "structural switch set per soil column, not a default "
                             "trainable closure",
        },
        "params": {
            "K_sat": {"units": "m s-1", "bounds": (9.537e-07, 8.67e-06), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
            "alpha_vg": {"units": "m-1", "bounds": (1.188, 10.8), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
            "b_ch": {"units": "1", "bounds": (1.7787, 16.17), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
            "c_film": {"units": "m5/2 s-1", "bounds": (4.455e-09, 4.05e-08), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
            "h0_pdi": {"units": "m", "bounds": (20790.0, 189000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
            "h_a": {"units": "m", "bounds": (33.0, 300.0), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
            "h_cav": {"units": "m", "bounds": (16.5, 150.0), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
            "h_crit": {"units": "m", "bounds": (0.0198, 0.18), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
            "lambda_bc": {"units": "1", "bounds": (0.06138, 0.558), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
            "n_a": {"units": "1", "bounds": (0.165, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
            "n_vg": {"units": "1", "bounds": (1.05, 4.0), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
            "omega_pdi": {"units": "1", "bounds": (0.165, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
            "psi_b": {"units": "m", "bounds": (-1.434, -0.15774), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
            "psi_sat": {"units": "m", "bounds": (-1.434, -0.15774), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
            "sigma_cav": {"units": "1", "bounds": (0.3, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
            "tau_s": {"units": "1", "bounds": (0.033, 0.3), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
            "theta_a": {"units": "1", "bounds": (0.0066, 0.06), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
            "theta_r": {"units": "1", "bounds": (0.02574, 0.234), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
            "theta_sat": {"units": "1", "bounds": (0.1419, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "van Genuchten / Clapp-Hornberger / Brooks-Corey", "shape": None},
        },
    },
}


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
    # Non-capillary (film) conductivity coefficient (Peters 2013)
    c_film: float = 1.35e-8     # [m^(5/2)/s]
    # Depth-decay of saturated conductivity: K_sat(z) = K_sat * exp(-z/k_sat_decay_m)
    # (Niu et al. 2005, J. Hydrometeorol.; soil compaction with depth).  Scales the
    # WHOLE K(theta) curve by exp(-z/k_sat_decay_m) at soil-node depth z, impeding
    # drainage OUT OF the root zone.  0.0 = disabled (uniform K, backward-compatible).
    k_sat_decay_m: float = 0.0  # [m] e-folding decay length; 0 => uniform with depth


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
    # AD-safe Mualem conductivity.  Two reachable boundaries carry an infinite
    # reverse-mode derivative even though the forward value is finite:
    #   * Se -> 0 (dry):       sqrt'(Se) = inf;
    #   * Se -> 1 (saturation, e.g. a ponded top layer via richards' K_top on
    #     raw psi): the inner term 1 - (1 - Se^{1/m})^m -> 1, but (1-Se^{1/m})^m
    #     has an infinite slope as its base u -> 0.
    # sqrt: floor Se at a sub-physical 1e-12 (Se=0 needs psi=-inf), so the
    # forward is bit-identical for any real Se.  Mualem term: a where-before-pow
    # keeps the forward *exact* at saturation (u=0 -> u**m=0 -> inner=1 ->
    # K=K_sat) — important for the K(psi=0)==K_sat contract — while masking the
    # pow's infinite-slope branch (the dead branch raises 1.0, not 0).
    Se_safe = jnp.maximum(Se, 1e-12)
    u = 1.0 - Se_safe ** (1.0 / m)
    mask = u > 1e-12
    u_pow = jnp.where(mask, jnp.where(mask, u, 1.0) ** m, 0.0)
    inner = 1.0 - u_pow
    return config.K_sat * jnp.sqrt(Se_safe) * inner ** 2


def van_genuchten_C(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Specific moisture capacity dtheta/dpsi."""
    m = 1.0 - 1.0 / config.n_vg
    n = config.n_vg
    alpha = config.alpha_vg
    # Floor |alpha*psi| away from 0 before the (n-1) power.  At psi -> 0
    # (saturation) |alpha psi|^{n-1} has an infinite derivative for n < 2 (the
    # usual VG case), so its reverse-mode gradient is inf there and the ``where``
    # below turns that into 0*inf = NaN — reachable whenever a layer ponds /
    # saturates (psi=0), which happens every infiltration event and is hit each
    # Picard iteration via moisture_capacity().  The floor is negligible
    # (|psi| ~ 1e-12/alpha) so the forward stays unchanged for any real psi.
    abs_alpha_psi = jnp.maximum(jnp.abs(alpha * psi), 1e-12)
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
    gamma_val = _GAMMA_VAL_OFFSET + 0.25 * Gamma_h0
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
    b0 = _B0_LOG10_COEFF * jnp.log(10.0)
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

    # Cosine interpolation between h_crit and h=0.
    # Boundary conditions: Kr_interp(h~0) = 1 (= K_sat) and
    # Kr_interp(h=h_crit) = Kr_crit (continuous match with capillary Kr_c).
    # frac = 0 at h~0 (wet), frac = 1 at h=h_crit (drier), so the smoothstep
    # must run from 1 at frac=0 to Kr_crit at frac=1.
    x = jnp.log10(jnp.clip(h, 1e-10, None))
    x_crit = jnp.log10(h_crit)
    x_s = jnp.log10(1e-10)  # effectively h ~ 0
    frac = jnp.clip((x - x_s) / (x_crit - x_s + _TINY), 0.0, 1.0)
    Kr_interp = 1.0 + 0.5 * (1.0 - jnp.cos(jnp.pi * frac)) * (Kr_crit - 1.0)

    # Use interpolated K when h < h_crit, otherwise standard capillary
    Kc = config.K_sat * jnp.where(h < h_crit, Kr_interp, Kr_c)

    # --- Non-capillary (film) conductivity ---
    # Knc = c_film * theta_r * (1 - Snc) * scale
    # Simplified: proportional to K_sat * (1 - Snc) * (theta_r/theta_sat)
    Snc = _pdi_Snc(h, config)
    c_film = config.c_film  # m^(5/2)/s (Peters 2013)
    ha = _pdi_ha(config)
    film_scale = ha ** (-1.5) - config.h0_pdi ** (-1.5)
    Knc = c_film * config.theta_r * jnp.clip(film_scale, 0.0, None) * (1.0 - Snc)

    K_total = Kc + Knc
    return jnp.where(psi >= 0.0, config.K_sat, K_total)


def pdi_C(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """PDI specific moisture capacity dtheta/dpsi (finite difference)."""
    eps = 1e-4  # coeff-ok: finite-difference / safety epsilon
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
    # where-before-pow guard (identical to van_genuchten_K): at saturation
    # Se=1 the Mualem base u = 1 - Se^(1/m) = 0, and u**m (m<1) has an infinite
    # slope, so the outer where(psi>=0, K_sat, K) would give 0*inf = NaN in the
    # reverse-mode gradient wrt theta_sat/theta_r.  Masking keeps that gradient
    # finite while the forward stays exact (u=0 -> inner=1 -> K_cap=K_sat).
    Se_safe = jnp.maximum(Se, 1e-12)
    u = 1.0 - Se_safe ** (1.0 / m)
    mask = u > 1e-12
    u_pow = jnp.where(mask, jnp.where(mask, u, 1.0) ** m, 0.0)
    inner = 1.0 - u_pow
    K_cap = config.K_sat * jnp.sqrt(Se_safe) * inner ** 2
    # Film-flow floor: prevents K=0 in the adsorptive regime
    K_film = config.K_sat * 1e-10
    K = jnp.maximum(K_cap, K_film)
    return jnp.where(psi >= 0.0, config.K_sat, K)


def lu_C(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Lu (2016) specific moisture capacity (finite difference)."""
    eps = 1e-4  # coeff-ok: finite-difference / safety epsilon
    theta_p = lu_theta(psi + eps, config)
    theta_m = lu_theta(psi - eps, config)
    return (theta_p - theta_m) / (2.0 * eps)


# ==========================================================================
# Dispatch functions
# ==========================================================================

def theta_from_psi(psi: jnp.ndarray, config: SoilHydraulicsConfig) -> jnp.ndarray:
    """Compute theta from psi using the configured retention curve.

    Above saturation (psi >= 0) the column stores additional water ELASTICALLY
    (specific storage): theta = theta_sat + S_s*theta_sat*psi.  This is the
    ParFlow / CliMA-Land "change of variable near saturation": psi is the primary
    variable everywhere and the capacity dtheta/dpsi = S_s*theta_sat stays > 0 at
    and above saturation, where the van-Genuchten capacity is 0.  It keeps the
    Richards matrix non-singular and theta CONSISTENT with ``moisture_capacity``,
    so the mass-conservative mixed form needs no non-physical theta clip (a clip
    at theta_sat silently destroyed the ponded/elastic storage) — ponding emerges
    as a positive head."""
    curve = config.retention_curve
    if curve == "van_genuchten":
        theta = van_genuchten_theta(psi, config)
    elif curve == "clapp_hornberger" or curve == "campbell":
        theta = clapp_hornberger_theta(psi, config)
    elif curve == "brooks_corey":
        theta = brooks_corey_theta(psi, config)
    elif curve == "pdi":
        theta = pdi_theta(psi, config)
    elif curve == "lu":
        theta = lu_theta(psi, config)
    else:
        raise ValueError(f"Unknown retention curve: {curve}")
    # Specific-storage branch (psi >= 0): the integral of the elastic capacity
    # S_s*theta_sat added in ``moisture_capacity``.  Zero below saturation so a
    # very dry psi never drives theta below theta_r.
    return theta + jnp.where(psi >= 0.0,
                             config.S_s * config.theta_sat * psi, 0.0)


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
    elif curve == "brooks_corey":
        # Brooks-Corey: finite-difference the RAW retention curve.  Using
        # theta_from_psi here would double-count the elastic S_s*theta_sat term
        # (theta_from_psi adds S_s*theta_sat*psi for psi>=0, and the block below
        # adds S_s*theta_sat again) -> C ~= 2*S_s*theta_sat at saturation,
        # inconsistent with dtheta/dpsi and the mixed-form Picard mass balance.
        # Every other scheme (van_genuchten_C, clapp_hornberger_C, pdi_C, lu_C)
        # differentiates the raw curve, so the S_s term is supplied exactly once.
        eps = 1e-4  # coeff-ok: finite-difference / safety epsilon
        theta_p = brooks_corey_theta(psi + eps, config)
        theta_m = brooks_corey_theta(psi - eps, config)
        C = (theta_p - theta_m) / (2.0 * eps)
    else:
        raise ValueError(f"Unknown retention curve: {curve}")

    # Elastic specific storage ABOVE saturation (psi >= 0) — the derivative of the
    # theta_from_psi specific-storage branch.  Conditional (not unconditional) so it
    # stays CONSISTENT with theta: d/dpsi[theta_sat + S_s*theta_sat*psi] = S_s*
    # theta_sat for psi>=0, and 0 below (where the van-Genuchten capacity governs).
    # This is what keeps C > 0 at saturation (van_genuchten_C -> 0 there) without
    # making theta inconsistent with C in the unsaturated zone (the mass-balance
    # error the mixed-form Picard would otherwise carry).
    C = C + jnp.where(psi >= 0.0, config.S_s * config.theta_sat, 0.0)
    return C


def interblock_K(
    K_above: jnp.ndarray,
    K_below: jnp.ndarray,
    log_f_above: jnp.ndarray | None = None,
    log_f_below: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Geometric mean of hydraulic conductivity between adjacent layers.

    Optional per-layer log multipliers (e.g. the frozen-soil ice impedance) are
    folded in log space, ``exp(0.5*(ln Ka + ln Kb + log_fa + log_fb))``, so a
    strongly impeded pair never forms the float32-subnormal product 1e-20*1e-20
    and reverse mode never differentiates sqrt at 0.  Without them the original
    sqrt(Ka*Kb) path runs unchanged.
    """
    Ka = jnp.clip(K_above, 1e-20, None)
    Kb = jnp.clip(K_below, 1e-20, None)
    if log_f_above is None and log_f_below is None:
        return jnp.sqrt(Ka * Kb)
    if log_f_above is None or log_f_below is None:
        raise ValueError("interblock_K: pass both log multipliers or neither.")
    return jnp.exp(0.5 * (jnp.log(Ka) + jnp.log(Kb) + log_f_above + log_f_below))


# ==========================================================================
# Per-column / per-layer parameter support
# ==========================================================================

def slice_layer(config: SoilHydraulicsConfig, k: int) -> SoilHydraulicsConfig:
    """Return ``config`` with every per-(col,layer) field reduced to a single
    layer ``k`` (so a ``(ncol, nlayer)`` or ``(ncol, 1)`` param becomes
    ``(ncol,)``).  Scalars and 1-D fields are passed through unchanged.

    Use this when a Richards step computes a quantity at a *single* layer
    (e.g. top-layer ``K_top`` for the infiltration capacity, bottom-layer
    ``K_bot`` for free-drainage runoff) and the soil state at that layer is
    ``(ncol,)``: mixing ``(ncol,)`` with a ``(ncol, 1)`` param would otherwise
    broadcast to ``(ncol, ncol)`` and silently corrupt the result.
    """
    def pick(v):
        # Strings (e.g. retention_curve) and Python scalars: passthrough.
        if not hasattr(v, "ndim"):
            return v
        # 0-D / 1-D arrays already align with (ncol,).
        if v.ndim < 2:
            return v
        # 2-D (ncol, n_layer_or_1): pick layer k.  Length-1 layer axis acts
        # as a broadcast and the index folds to 0 automatically.
        idx = k if v.shape[-1] > 1 else 0
        return v[..., idx]
    return type(config)(*(pick(getattr(config, f)) for f in config._fields))
