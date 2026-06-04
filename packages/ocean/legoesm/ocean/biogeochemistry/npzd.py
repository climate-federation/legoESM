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
    # Shift: interface attenuation at top of layer k = cumsum through k-1.
    # ``jnp.pad`` is one Pad HLO op; the previous form allocated a fresh
    # ``(..., 1)`` zero buffer + concatenate.
    _pad_axes = ((0, 0),) * (cum_atten_interface.ndim - 1)
    cum_atten_top = jnp.pad(cum_atten_interface[..., :-1], (*_pad_axes, (1, 0)))
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

    # Light limitation (Webb et al. 1974 exponential form)
    # alpha_P [1/(W/m^2)/day] and mu_max [1/day] are both per-day,
    # so their ratio alpha_P/mu_max [1/(W/m^2)] is already consistent.
    f_L = 1.0 - jnp.exp(-cfg.alpha_P * PAR / jnp.clip(cfg.mu_max, eps, None))

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

    # ---- 5. Detritus sinking (conservative interface-flux formulation) ----
    # w_sink in m/day -> m/s
    w_sink_s = cfg.w_sink * day_to_s  # m/s
    # Interface flux (upwind): F[k] = w_sink * D[k-1] [mol N/m^2/s]
    # F[0] = 0 (no flux into top), F[nlev] = w_sink * D[nlev-1] (export)
    # Tendency: dD/dt[k] = (F[k] - F[k+1]) / dz[k]
    flux_out = w_sink_s * D  # flux leaving each layer downward
    # ``jnp.pad`` along trailing axis: single Pad HLO op vs
    # alloc-zeros + concatenate.
    _pad_axes_d = ((0, 0),) * (flux_out.ndim - 1)
    flux_in = jnp.pad(flux_out[..., :-1], (*_pad_axes_d, (1, 0)))
    sinking_tend = (flux_in - flux_out) / jnp.clip(dz_ref, 1.0, None)

    # ---- Assemble tendencies ----
    # Nutrients
    dNO3_dt = -growth + remin + (1.0 - cfg.gamma_Z) * grazing

    # Phytoplankton
    dPhyto_dt = growth - grazing - phyto_mort

    # Zooplankton
    dZoo_dt = cfg.gamma_Z * grazing - zoo_mort

    # Detritus
    dDet_dt = phyto_mort + zoo_mort - remin + sinking_tend

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
    # Alkalinity (Dickson total alkalinity, which carries the
    # ``-[NO3-]`` term):
    #   * NO3 uptake by phytoplankton REMOVES nitrate from solution, so
    #     -d[NO3-]/dt is positive ⇒ TA increases ⇒ +growth.
    #   * Remineralization adds NO3- back ⇒ TA decreases ⇒ -remin.
    #   * Zooplankton excretion (``(1-gamma_Z)*grazing``) returns N as
    #     NO3- to solution ⇒ -(1-gamma_Z)*grazing.
    #   * CaCO3 precipitation removes 2 mol of charge per mol CaCO3
    #     ⇒ -2*caco3_production.
    #   * CaCO3 dissolution adds 2 mol of charge ⇒ +2*caco3_dissolution.
    # The previous code had the N-cycle signs flipped (was tracking
    # dNO3_dt instead of -dNO3_dt for the organic terms) — caught by
    # codex adversarial review (iter-1).
    dALK_dt = (growth - remin - (1.0 - cfg.gamma_Z) * grazing
               - 2.0 * caco3_production + 2.0 * caco3_dissolution)

    return dNO3_dt, dPhyto_dt, dZoo_dt, dDet_dt, dDIC_dt, dALK_dt
