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


# ======================================================================= #
#  npzd_v2 — 18-tracer source/sink (2 PFTs, NH4, PO4/Si/Fe, DON/DOP,     #
#  POC/DOC, O2).  Ported from ggn/bgc_modelling; additive to v1 above.    #
# ======================================================================= #

def par_profile_v2(
    PAR_surf: jnp.ndarray,
    Chl_d: jnp.ndarray,
    Chl_n: jnp.ndarray,
    dz_ref: jnp.ndarray,
    cfg: BiogeoConfig,
) -> jnp.ndarray:
    """PAR profile using total (diatom + nanophyto) Chl self-shading."""
    Chl_tot = jnp.clip(Chl_d, 0.0, None) + jnp.clip(Chl_n, 0.0, None)
    atten = (cfg.k_w_atten + cfg.k_chl_atten * Chl_tot) * dz_ref
    cum_interface = jnp.cumsum(atten, axis=-1)
    _pad = ((0, 0),) * (atten.ndim - 1)
    cum_top = jnp.pad(cum_interface[..., :-1], (*_pad, (1, 0)))
    cum_mid = cum_top + 0.5 * atten
    return PAR_surf[..., jnp.newaxis] * jnp.exp(-cum_mid)


def npzd_v2_source_sink(
    NO3, NH4, PO4, Si, Fe,
    Pd, Pn, Chl_d, Chl_n,
    Zoo, Det, DON, DOP,
    DIC, POC, DOC, ALK, O2,
    T_degC: jnp.ndarray,
    PAR: jnp.ndarray,
    dz_ref: jnp.ndarray,
    cfg: BiogeoConfig,
) -> tuple[jnp.ndarray, ...]:
    """18-tracer source/sink terms for npzd_v2.

    Returns (dNO3, dNH4, dPO4, dSi, dFe, dPd, dPn, dChl_d, dChl_n,
             dZoo, dDet, dDON, dDOP, dDIC, dPOC, dDOC, dALK, dO2),
    all [mol/m^3/s] ([kg Chl/m^3/s] for Chl).  Settling of Det is an upwind
    flux over ``dz_ref`` inside this function (to move into the transport
    operator later, ShyBFM-style).  The aeolian iron source is NOT here: the
    host adds ``cfg.Fe_dust / dz_surface`` to the surface level.
    """
    day_s = 1.0 / 86400.0
    eps = 1.0e-30

    NO3_ = jnp.clip(NO3, 0.0, None);  NH4_ = jnp.clip(NH4, 0.0, None)
    PO4_ = jnp.clip(PO4, 0.0, None);  Si_ = jnp.clip(Si, 0.0, None)
    Fe_ = jnp.clip(Fe, 0.0, None)
    Pd_ = jnp.clip(Pd, 0.0, None);  Pn_ = jnp.clip(Pn, 0.0, None)
    Cd_ = jnp.clip(Chl_d, 0.0, None);  Cn_ = jnp.clip(Chl_n, 0.0, None)
    Z_ = jnp.clip(Zoo, 0.0, None);  D_ = jnp.clip(Det, 0.0, None)
    DON_ = jnp.clip(DON, 0.0, None);  DOP_ = jnp.clip(DOP, 0.0, None)
    POC_ = jnp.clip(POC, 0.0, None);  DOC_ = jnp.clip(DOC, 0.0, None)
    O2_ = jnp.clip(O2, 0.0, None)

    # ---- temperature ----
    Tq = 1.066 ** jnp.clip(T_degC, -2.0, 40.0)
    T_fac_d = (cfg.mu_max_d * day_s) * Tq
    T_fac_n = (cfg.mu_max_n * day_s) * Tq

    # ---- NH4 inhibition of NO3 uptake ----
    psi = jnp.exp(-cfg.psi_coeff * NH4_)

    # ---- limitation ----
    gNO3_d = NO3_ / (cfg.k_N_d + NO3_) * psi
    gNH4_d = NH4_ / (cfg.k_NH4_d + NH4_)
    gN_d = gNO3_d + gNH4_d
    fNH4_d = gNH4_d / (gN_d + eps)
    gNO3_n = NO3_ / (cfg.k_N_n + NO3_) * psi
    gNH4_n = NH4_ / (cfg.k_NH4_n + NH4_)
    gN_n = gNO3_n + gNH4_n
    fNH4_n = gNH4_n / (gN_n + eps)
    gI = 1.0 - jnp.exp(-cfg.alpha_P_v2 * PAR / jnp.clip(cfg.mu_max_d, eps, None))
    gFe_d = Fe_ / (cfg.k_Fe_d + Fe_)
    gFe_n = Fe_ / (cfg.k_Fe_n + Fe_)
    gSi = Si_ / (cfg.k_Si + Si_)
    mu_d = T_fac_d * jnp.minimum(gN_d, 1.0) * gI * gFe_d * gSi
    mu_n = T_fac_n * jnp.minimum(gN_n, 1.0) * gI * gFe_n

    # ---- grazing (equal preference, Holling II on total prey) ----
    prey = Pd_ + Pn_
    G_tot = (cfg.g_max * day_s) * prey / (cfg.k_P + prey)
    G_d = G_tot * Pd_ / (prey + eps)
    G_n = G_tot * Pn_ / (prey + eps)
    # Total grazing in mol N/m^3/s: G_d, G_n are rates [1/s] already weighted
    # by the prey fractions, and the phyto losses below are G_d*Z, G_n*Z.
    # (The branch version multiplied by P again -> zooplankton never fed and
    # N leaked at the grazing rate; found by the column N budget.)
    G_tot_N = (G_d + G_n) * Z_
    egestion = (1.0 - cfg.epse) * (1.0 - cfg.gamma_Z) * G_tot_N     # -> Det
    excretion = cfg.epse * (1.0 - cfg.gamma_Z) * G_tot_N            # -> NH4

    # ---- nitrification ----
    nitrif = (cfg.k_nitrif * day_s) * NH4_ * jnp.exp(-cfg.kI_nitrif * PAR)

    # ---- phytoplankton ----
    mpl_s = cfg.m_Pl * day_s
    mpq_s = cfg.m_Pq_v2 * day_s
    dPd = mu_d * Pd_ - G_d * Z_ - mpl_s * Pd_ - mpq_s * Pd_ ** 2
    dPn = mu_n * Pn_ - G_n * Z_ - mpl_s * Pn_ - mpq_s * Pn_ ** 2

    # ---- chlorophyll: target Chl:N follows light, Chl tracks biomass ----
    theta_tgt_d = cfg.theta_min_d + (cfg.theta_max_d - cfg.theta_min_d) * (1.0 - gI)
    theta_tgt_n = cfg.theta_min_n + (cfg.theta_max_n - cfg.theta_min_n) * (1.0 - gI)
    dChl_d = theta_tgt_d * (mu_d - G_d * Z_ / jnp.maximum(Pd_, 1e-15) - mpl_s - mpq_s * Pd_) * Pd_
    dChl_n = theta_tgt_n * (mu_n - G_n * Z_ / jnp.maximum(Pn_, 1e-15) - mpl_s - mpq_s * Pn_) * Pn_

    # ---- zooplankton ----
    mzl_s = cfg.m_Zl_v2 * day_s
    mzq_s = cfg.m_Z * day_s
    dZoo = cfg.gamma_Z * G_tot_N - mzl_s * Z_ - mzq_s * Z_ ** 2

    # ---- detritus (+ upwind settling over dz_ref) ----
    remin_s = cfg.remin_rate * day_s
    dDet_bio = (mpl_s * (Pd_ + Pn_) + mpq_s * (Pd_ ** 2 + Pn_ ** 2)
                + mzq_s * Z_ ** 2 + egestion - remin_s * D_)
    w_s = cfg.w_sink * day_s
    flux_out = w_s * D_
    _pad = ((0, 0),) * (D_.ndim - 1)
    flux_in = jnp.pad(flux_out[..., :-1], (*_pad, (1, 0)))
    sink_tend = (flux_in - flux_out) / jnp.clip(dz_ref, 1.0, None)
    # bottom flux returned instantly as NH4 (zero-storage benthic return; closes N)
    bottom_flux = flux_out[..., -1] / jnp.clip(dz_ref[-1], 1.0, None)
    bottom_return = jnp.pad(bottom_flux[..., jnp.newaxis], (*_pad, (D_.shape[-1] - 1, 0)))
    dDet = dDet_bio + sink_tend

    # ---- DON / DOP ----
    remin_DON_s = cfg.remin_DON * day_s
    dDON = mzl_s * Z_ - remin_DON_s * DON_
    dDOP = dDON / cfg.R_NP

    # ---- inorganic N ----
    mu_tot = mu_d * Pd_ + mu_n * Pn_
    mu_NH4 = mu_d * Pd_ * fNH4_d + mu_n * Pn_ * fNH4_n
    mu_NO3 = mu_tot - mu_NH4
    regen = remin_s * D_ + remin_DON_s * DON_ + excretion
    dNH4 = -mu_NH4 - nitrif + regen + bottom_return
    dNO3 = -mu_NO3 + nitrif
    dPO4 = (dNO3 + dNH4) / cfg.R_NP

    # ---- Si ----
    remin_opal_s = cfg.remin_opal * day_s
    dSi = -cfg.R_SiN * mu_d * Pd_ + cfg.R_SiN * remin_opal_s * D_

    # ---- Fe (prognostic; dust added by the host at the surface) ----
    Fe_uptake = cfg.R_FeN_d * mu_d * Pd_ + cfg.R_FeN_n * mu_n * Pn_
    Fe_regen = cfg.R_FeN_avg * (remin_s * D_ + remin_DON_s * DON_)
    Fe_scav = (cfg.k_Fe_scav * day_s) * jnp.maximum(Fe_ - cfg.Fe_ligand, 0.0)
    dFe = -Fe_uptake + Fe_regen - Fe_scav

    # ---- POC / DOC ----
    remin_DOC_s = cfg.remin_DOC * day_s
    phyto_mort_tot = mpl_s * (Pd_ + Pn_)
    dPOC = (cfg.f_POC * phyto_mort_tot + mzq_s * Z_ ** 2 + egestion - remin_s * POC_)
    dDOC = ((1.0 - cfg.f_POC) * phyto_mort_tot * cfg.R_CN - remin_DOC_s * DOC_)

    # ---- DIC / ALK ----
    dDIC = -cfg.R_CN * mu_tot + cfg.R_CN * regen - remin_DOC_s * DOC_
    dALK = mu_NO3 - mu_NH4 - 2.0 * nitrif

    # ---- O2: photosynthesis +R_ON per N fixed, respiration -R_ON per N regenerated,
    #      nitrification -2 O2 per N; limited to O2 > 0 by the host's positivity ----
    dO2 = cfg.R_ON * (mu_tot - regen) - 2.0 * nitrif

    return (dNO3, dNH4, dPO4, dSi, dFe,
            dPd, dPn, dChl_d, dChl_n,
            dZoo, dDet, dDON, dDOP,
            dDIC, dPOC, dDOC, dALK, dO2)
