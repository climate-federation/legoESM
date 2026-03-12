"""NPZD ecosystem model coupled to the inorganic carbon cycle.

A Nutrient-Phytoplankton-Zooplankton-Detritus model following the
Fasham et al. (1990) / Oschlies & Garçon (1999) framework with
coupling to DIC, alkalinity, and air-sea CO2 exchange.

Processes:
1. Light-limited, nutrient-limited phytoplankton growth
2. Zooplankton grazing (Holling type II)
3. Phytoplankton and zooplankton mortality
4. Detritus sinking and remineralization
5. Stoichiometric coupling to DIC/ALK via Redfield ratios
6. CaCO3 production/dissolution (rain ratio parameterization)

All functions are JAX-compatible (differentiable, JIT-friendly).

References
----------
- Fasham, M. J. R., et al. (1990). A nitrogen-based model of plankton
  dynamics in the oceanic mixed layer. J. Mar. Res., 48, 591-639.
- Oschlies, A. & Garçon, V. (1999). An eddy-permitting coupled physical-
  biological model of the North Atlantic. Global Biogeochem. Cycles, 13.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.biogeochemistry.config import BiogeoConfig


def par_profile(
    PAR_surf: jnp.ndarray,
    z_full_ref: jnp.ndarray,
    Phyto: jnp.ndarray,
    dz_ref: jnp.ndarray,
    cfg: BiogeoConfig,
) -> jnp.ndarray:
    """Compute PAR at each depth level with self-shading.

    Beer-Lambert law with water + chlorophyll attenuation:
    PAR(z) = PAR_surf * exp(-integral_0^z (k_w + k_chl * P) dz')

    Parameters
    ----------
    PAR_surf : array
        Surface PAR [W/m^2], shape (...).
    z_full_ref : array
        Reference depths [m], shape (nlev,), negative.
    Phyto : array
        Phytoplankton [mol N/m^3], shape (..., nlev).
    dz_ref : array
        Layer thicknesses [m], shape (nlev,).
    cfg : BiogeoConfig

    Returns
    -------
    PAR : array
        PAR at each level [W/m^2], shape (..., nlev).
    """
    nlev = z_full_ref.shape[0]

    # Total attenuation per layer: (k_w + k_chl * P) * dz
    atten = (cfg.k_w_atten + cfg.k_chl_atten * jnp.clip(Phyto, 0.0, None)) * dz_ref

    # Cumulative attenuation from surface (k=0 is surface)
    # For level k, PAR has been attenuated by layers 0..k-1 plus half of layer k
    cum_atten_interface = jnp.cumsum(atten, axis=-1)
    # Shift: interface attenuation at top of layer k = cumsum through k-1
    zeros = jnp.zeros((*atten.shape[:-1], 1), dtype=atten.dtype)
    cum_atten_top = jnp.concatenate([zeros, cum_atten_interface[..., :-1]], axis=-1)
    # Mid-level attenuation = top + half this layer
    cum_atten_mid = cum_atten_top + 0.5 * atten

    PAR = PAR_surf[..., jnp.newaxis] * jnp.exp(-cum_atten_mid)
    return PAR


def npzd_source_sink(
    NO3: jnp.ndarray,
    Phyto: jnp.ndarray,
    Zoo: jnp.ndarray,
    Det: jnp.ndarray,
    DIC: jnp.ndarray,
    ALK: jnp.ndarray,
    T_degC: jnp.ndarray,
    PAR: jnp.ndarray,
    dz_ref: jnp.ndarray,
    cfg: BiogeoConfig,
) -> tuple[jnp.ndarray, ...]:
    """Compute NPZD source/sink terms at every grid point and level.

    Parameters
    ----------
    NO3, Phyto, Zoo, Det, DIC, ALK : array (..., nlev)
        Biogeochemical tracer concentrations.
    T_degC : array (..., nlev)
        Temperature [degC].
    PAR : array (..., nlev)
        Photosynthetically available radiation [W/m^2].
    dz_ref : array (nlev,)
        Layer thicknesses [m].
    cfg : BiogeoConfig

    Returns
    -------
    dNO3_dt, dPhyto_dt, dZoo_dt, dDet_dt, dDIC_dt, dALK_dt : arrays
        Source/sink tendencies [units/s], same shapes as inputs.
    """
    day_to_s = 1.0 / 86400.0  # convert rates from 1/day to 1/s

    # Clip tracers to non-negative (smooth via softplus at small values)
    eps = 1.0e-12
    P = jnp.clip(Phyto, eps, None)
    Z = jnp.clip(Zoo, eps, None)
    D = jnp.clip(Det, eps, None)
    N = jnp.clip(NO3, eps, None)

    # ---- 1. Phytoplankton growth ----
    # Temperature dependence: Eppley (1972) Q10 = 1.066^T
    T_factor = 1.066 ** jnp.clip(T_degC, -2.0, 40.0)

    # Nutrient limitation (Michaelis-Menten)
    f_N = N / (N + cfg.k_N)

    # Light limitation (Smith 1936 hyperbolic tangent)
    f_L = 1.0 - jnp.exp(-cfg.alpha_P * PAR / jnp.clip(cfg.mu_max * day_to_s, eps, None))

    # Growth rate
    mu = cfg.mu_max * day_to_s * T_factor * jnp.minimum(f_N, f_L)
    growth = mu * P  # mol N/m^3/s

    # ---- 2. Zooplankton grazing ----
    grazing = cfg.g_max * day_to_s * P ** 2 / (P ** 2 + cfg.k_P ** 2) * Z

    # ---- 3. Mortality ----
    phyto_mort = cfg.m_P * day_to_s * P
    zoo_mort = cfg.m_Z * day_to_s * Z ** 2  # quadratic

    # ---- 4. Detritus remineralization ----
    remin = cfg.remin_rate * day_to_s * D

    # ---- 5. Detritus sinking (as a loss from each layer, gain to layer below) ----
    # w_sink in m/day -> m/s
    w_sink_s = cfg.w_sink * day_to_s  # m/s
    # Sinking flux out of layer k: w_sink * D[k] / dz[k]
    # This is a simple first-order upwind treatment
    sink_loss = w_sink_s * D / jnp.clip(dz_ref, 1.0, None)
    # Gain from layer above (k-1): shift and zero at surface
    zeros = jnp.zeros((*D.shape[:-1], 1), dtype=D.dtype)
    sink_from_above = jnp.concatenate([zeros, (w_sink_s * D / jnp.clip(dz_ref, 1.0, None))[..., :-1]], axis=-1)

    # ---- Assemble tendencies ----
    # Nutrients
    dNO3_dt = -growth + remin + (1.0 - cfg.gamma_Z) * grazing

    # Phytoplankton
    dPhyto_dt = growth - grazing - phyto_mort

    # Zooplankton
    dZoo_dt = cfg.gamma_Z * grazing - zoo_mort

    # Detritus
    dDet_dt = phyto_mort + zoo_mort + (1.0 - cfg.gamma_Z) * grazing - grazing * 0  # sloppy feed already in NO3
    dDet_dt = phyto_mort + zoo_mort - remin - sink_loss + sink_from_above

    # ---- Carbon coupling (Redfield) ----
    # Organic carbon cycle: C:N = R_CN
    # DIC decreases with primary production, increases with remineralization
    net_production = growth - remin - (1.0 - cfg.gamma_Z) * grazing
    dDIC_bio = -cfg.R_CN * net_production

    # CaCO3 cycle: rain ratio * organic C export
    # CaCO3 production removes DIC and 2*ALK (in surface/euphotic zone)
    # CaCO3 dissolution adds back (in deep, parameterized as remin)
    caco3_production = cfg.R_CaP * cfg.R_CN * growth
    caco3_dissolution = cfg.R_CaP * cfg.R_CN * remin

    dDIC_dt = dDIC_bio - caco3_production + caco3_dissolution
    # Alkalinity: -1 per mol NO3 consumed (nitrification sign convention)
    # + 2 per mol CaCO3 dissolved, -2 per mol CaCO3 precipitated
    dALK_dt = (-growth + remin + (1.0 - cfg.gamma_Z) * grazing
               - 2.0 * caco3_production + 2.0 * caco3_dissolution)

    return dNO3_dt, dPhyto_dt, dZoo_dt, dDet_dt, dDIC_dt, dALK_dt
