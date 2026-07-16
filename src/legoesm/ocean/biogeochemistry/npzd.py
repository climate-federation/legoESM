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

    # Growth rate — multiplicative limitation (Fasham 1990)
    mu = cfg.mu_max * day_to_s * T_factor * f_N * f_L
    growth = mu * P  # mol N/m^3/s

    # ---- 2. Zooplankton grazing (Holling type II) ----
    grazing = cfg.g_max * day_to_s * P / (P + cfg.k_P) * Z

    # ---- 3. Mortality ----
    # Phytoplankton: linear (senescence) + quadratic (aggregation/export)
    phyto_mort = cfg.m_P  * day_to_s * P
    phyto_agg  = cfg.m_Pq * day_to_s * P ** 2
    # Zooplankton: linear (starvation/predation) + quadratic (higher predation)
    zoo_mort_l = cfg.m_Zl * day_to_s * Z
    zoo_mort_q = cfg.m_Z  * day_to_s * Z ** 2
    zoo_mort   = zoo_mort_l + zoo_mort_q

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

    # ---- Assemble N-cycle tendencies (mass-conserving) ----
    # Sloppy feeding / egestion: fraction (1-epse) of non-assimilated grazing
    # goes to detritus; fraction epse goes directly to N (excretion).
    egestion    = (1.0 - cfg.epse) * (1.0 - cfg.gamma_Z) * grazing  # → Det
    excretion   = cfg.epse         * (1.0 - cfg.gamma_Z) * grazing  # → N

    # Phytoplankton: linear mort goes directly to N (dissolved), quadratic to Det
    dPhyto_dt = growth - grazing - phyto_mort - phyto_agg

    # Zooplankton: assimilation gain; linear mort → N, quadratic → Det
    dZoo_dt = cfg.gamma_Z * grazing - zoo_mort

    # Detritus biological source/sink (no transport)
    dDet_bio = phyto_agg + zoo_mort_q + egestion - remin
    # Full detritus tendency including sinking transport
    dDet_dt = dDet_bio + sinking_tend

    # Nutrient: mass conservation for *biological* terms only.
    # Sinking is a transport term, not a N source. The bottom sinking flux
    # (export) is returned to NO3 via instant sediment remineralization,
    # closing the N cycle without a full sediment model.
    bottom_flux = w_sink_s * D[..., -1] / jnp.clip(dz_ref[-1], 1.0, None)
    _pad_axes_b = ((0, 0),) * (D.ndim - 1)
    bottom_remin = jnp.pad(bottom_flux[..., jnp.newaxis],
                           (*_pad_axes_b, (D.shape[-1] - 1, 0)))
    dNO3_dt = -(dPhyto_dt + dZoo_dt + dDet_bio) + bottom_remin

    # ---- Carbon coupling (Redfield) ----
    # Net organic carbon production = growth − remineralization − N returned to solution
    # dNO3_dt (bio only, without sinking) captures this via mass conservation.
    # For DIC/ALK we need the bio-only dNO3 (exclude sinking which doesn't change DIC).
    dNO3_bio = -(dPhyto_dt + dZoo_dt + (phyto_agg + zoo_mort_q + egestion - remin))
    dDIC_bio  = -cfg.R_CN * dNO3_bio  # DIC mirrors N with Redfield ratio

    # CaCO3 cycle: rain ratio * organic C export
    # CaCO3 production removes DIC and 2*ALK (in surface/euphotic zone)
    # CaCO3 dissolution adds back (in deep, parameterized as remin)
    caco3_production = cfg.R_CaP * cfg.R_CN * growth
    caco3_dissolution = cfg.R_CaP * cfg.R_CN * remin

    dDIC_dt = dDIC_bio - caco3_production + caco3_dissolution
    # Alkalinity tracks -dNO3 (organic N terms only) ± 2×CaCO3.
    # dNO3_bio already encodes the sign-correct organic nitrogen cycle.
    dALK_dt = (-dNO3_bio
               - 2.0 * caco3_production + 2.0 * caco3_dissolution)

    return dNO3_dt, dPhyto_dt, dZoo_dt, dDet_dt, dDIC_dt, dALK_dt


# ======================================================================= #
#  npzd_v2 — 18-tracer source/sink (2 PFTs, NH4, POC/DOC, DOP, Fe, Si)   #
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
    atten   = (cfg.k_w_atten + cfg.k_chl_atten * Chl_tot) * dz_ref
    cum_interface = jnp.cumsum(atten, axis=-1)
    _pad = ((0, 0),) * (atten.ndim - 1)
    cum_top = jnp.pad(cum_interface[..., :-1], (*_pad, (1, 0)))
    cum_mid = cum_top + 0.5 * atten
    return PAR_surf[..., jnp.newaxis] * jnp.exp(-cum_mid)


def npzd_v2_source_sink(
    NO3: jnp.ndarray,
    NH4: jnp.ndarray,
    PO4: jnp.ndarray,
    Si: jnp.ndarray,
    Fe: jnp.ndarray,
    Pd: jnp.ndarray,
    Pn: jnp.ndarray,
    Chl_d: jnp.ndarray,
    Chl_n: jnp.ndarray,
    Zoo: jnp.ndarray,
    Det: jnp.ndarray,
    DON: jnp.ndarray,
    DOP: jnp.ndarray,
    DIC: jnp.ndarray,
    POC: jnp.ndarray,
    DOC: jnp.ndarray,
    ALK: jnp.ndarray,
    T_degC: jnp.ndarray,
    PAR: jnp.ndarray,
    dz_ref: jnp.ndarray,
    cfg: BiogeoConfig,
) -> tuple[jnp.ndarray, ...]:
    """Compute 18-tracer source/sink terms for npzd_v2.

    Returns (in order):
        dNO3, dNH4, dPO4, dSi, dFe,
        dPd, dPn, dChl_d, dChl_n,
        dZoo, dDet, dDON, dDOP,
        dDIC, dPOC, dDOC, dALK
    All in [mol/m^3/s] (or [kg Chl/m^3/s] for Chl).
    """
    day_s = 1.0 / 86400.0
    eps   = 1.0e-30

    # ---- Clip to non-negative ----
    NO3_ = jnp.clip(NO3, 0.0, None);  NH4_ = jnp.clip(NH4, 0.0, None)
    PO4_ = jnp.clip(PO4, 0.0, None);  Si_  = jnp.clip(Si,  0.0, None)
    Fe_  = jnp.clip(Fe,  0.0, None)
    Pd_  = jnp.clip(Pd,  0.0, None);  Pn_  = jnp.clip(Pn,  0.0, None)
    Cd_  = jnp.clip(Chl_d, 0.0, None); Cn_ = jnp.clip(Chl_n, 0.0, None)
    Z_   = jnp.clip(Zoo, 0.0, None);  D_   = jnp.clip(Det,  0.0, None)
    DON_ = jnp.clip(DON, 0.0, None);  DOP_ = jnp.clip(DOP,  0.0, None)
    POC_ = jnp.clip(POC, 0.0, None);  DOC_ = jnp.clip(DOC,  0.0, None)

    # ---- Temperature ----
    T_fac_d = (cfg.mu_max_d * day_s) * (1.066 ** jnp.clip(T_degC, -2.0, 40.0))
    T_fac_n = (cfg.mu_max_n * day_s) * (1.066 ** jnp.clip(T_degC, -2.0, 40.0))

    # ---- NH4 inhibition of NO3 uptake ----
    psi = jnp.exp(-cfg.psi_coeff * NH4_)    # [0,1]

    # ---- Limitation factors ----
    gNO3_d = NO3_ / (cfg.k_N_d + NO3_) * psi
    gNH4_d = NH4_ / (cfg.k_NH4_d + NH4_)
    gN_d   = gNO3_d + gNH4_d
    fNH4_d = gNH4_d / (gN_d + eps)

    gNO3_n = NO3_ / (cfg.k_N_n + NO3_) * psi
    gNH4_n = NH4_ / (cfg.k_NH4_n + NH4_)
    gN_n   = gNO3_n + gNH4_n
    fNH4_n = gNH4_n / (gN_n + eps)

    # gI: alpha_P in (W/m2)^-1 day^-1, mu_max_d in day^-1 — keep consistent units
    gI   = 1.0 - jnp.exp(-cfg.alpha_P * PAR /
                          jnp.clip(cfg.mu_max_d, eps, None))
    gFe_d = Fe_ / (cfg.k_Fe_d + Fe_)
    gFe_n = Fe_ / (cfg.k_Fe_n + Fe_)
    gSi   = Si_ / (cfg.k_Si   + Si_)

    mu_d = T_fac_d * jnp.minimum(gN_d, 1.0) * gI * gFe_d * gSi
    mu_n = T_fac_n * jnp.minimum(gN_n, 1.0) * gI * gFe_n

    # ---- Grazing (equal preference) ----
    prey  = Pd_ + Pn_
    G_tot = (cfg.g_max * day_s) * prey / (cfg.k_P + prey)
    G_d   = G_tot * Pd_ / (prey + eps)
    G_n   = G_tot * Pn_ / (prey + eps)
    G_tot_N  = (G_d * Pd_ + G_n * Pn_) * Z_
    egestion  = (1.0 - cfg.epse) * (1.0 - cfg.gamma_Z) * G_tot_N   # → Det
    excretion =        cfg.epse  * (1.0 - cfg.gamma_Z) * G_tot_N   # → NH4

    # ---- Nitrification ----
    nitrif = (cfg.k_nitrif * day_s) * NH4_ * jnp.exp(-cfg.kI_nitrif * PAR)

    # ---- Phyto ----
    mpl_s  = cfg.m_Pl   * day_s
    mpq_s  = cfg.m_Pq_v2 * day_s

    dPd = mu_d*Pd_ - G_d*Z_ - mpl_s*Pd_ - mpq_s*Pd_**2
    dPn = mu_n*Pn_ - G_n*Z_ - mpl_s*Pn_ - mpq_s*Pn_**2

    # ---- Chlorophyll (prognostic Geider) ----
    theta_tgt_d = cfg.theta_min_d + (cfg.theta_max_d - cfg.theta_min_d) * (1.0 - gI)
    theta_tgt_n = cfg.theta_min_n + (cfg.theta_max_n - cfg.theta_min_n) * (1.0 - gI)
    # theta_cur clipped to [theta_min, theta_max] to prevent runaway Chl:N
    theta_cur_d = jnp.where(Pd_ > 1.0e-12,
        jnp.clip(Cd_ / Pd_, cfg.theta_min_d, cfg.theta_max_d),
        theta_tgt_d)
    theta_cur_n = jnp.where(Pn_ > 1.0e-12,
        jnp.clip(Cn_ / Pn_, cfg.theta_min_n, cfg.theta_max_n),
        theta_tgt_n)
    # Chl: diagnostic — set Chl = theta_tgt * P at each step
    # This avoids the ill-conditioned prognostic Chl equation
    # and is equivalent to assuming instantaneous Chl:N adaptation
    dChl_d = theta_tgt_d * (mu_d - G_d*Z_/jnp.maximum(Pd_, 1e-15) - mpl_s - mpq_s*Pd_) * Pd_
    dChl_n = theta_tgt_n * (mu_n - G_n*Z_/jnp.maximum(Pn_, 1e-15) - mpl_s - mpq_s*Pn_) * Pn_

    # ---- Zoo ----
    mzl_s = cfg.m_Zl_v2 * day_s
    mzq_s = cfg.m_Z     * day_s
    dZoo  = cfg.gamma_Z * G_tot_N - mzl_s*Z_ - mzq_s*Z_**2

    # ---- Detritus (= Det in v1) ----
    remin_s = cfg.remin_rate * day_s
    dDet_bio = (mpl_s*(Pd_ + Pn_) + mpq_s*(Pd_**2 + Pn_**2)
                + mzq_s*Z_**2 + egestion - remin_s*D_)

    # Sinking (upwind)
    w_s     = cfg.w_sink * day_s
    flux_out = w_s * D_
    _pad     = ((0, 0),) * (D_.ndim - 1)
    flux_in  = jnp.pad(flux_out[..., :-1], (*_pad, (1, 0)))
    sink_tend = (flux_in - flux_out) / jnp.clip(dz_ref, 1.0, None)
    dDet = dDet_bio + sink_tend

    # ---- DON / DOP ----
    remin_DON_s = cfg.remin_DON * day_s
    dDON = mzl_s * Z_ - remin_DON_s * DON_
    dDOP = dDON / cfg.R_NP

    # ---- NH4 ----
    mu_tot  = mu_d*Pd_ + mu_n*Pn_
    mu_NH4  = mu_d*Pd_*fNH4_d + mu_n*Pn_*fNH4_n
    mu_NO3  = mu_tot - mu_NH4
    dNH4 = (-mu_NH4 - nitrif + remin_s*D_ + remin_DON_s*DON_ + excretion)

    # ---- NO3 ----
    dNO3 = -mu_NO3 + nitrif

    # ---- PO4 ----
    dPO4 = (dNO3 + dNH4) / cfg.R_NP

    # ---- Si ----
    remin_opal_s = cfg.remin_opal * day_s
    dSi = -cfg.R_SiN * mu_d * Pd_ + cfg.R_SiN * remin_opal_s * D_

    # ---- Fe ----
    Fe_uptake = cfg.R_FeN_d * mu_d * Pd_ + cfg.R_FeN_n * mu_n * Pn_
    Fe_regen  = cfg.R_FeN_avg * (remin_s*D_ + remin_DON_s*DON_)
    Fe_scav   = (cfg.k_Fe_scav * day_s) * jnp.maximum(Fe_ - cfg.Fe_ligand, 0.0)
    dFe       = -Fe_uptake + Fe_regen - Fe_scav
    # Note: aeolian dust is added as a surface boundary condition in the
    # host model (cfg.Fe_dust [mol Fe/m^2/day] divided by dz at surface).

    # ---- POC / DOC ----
    # POC: phyto linear mort (f_POC fraction), zoo quadratic mort, egestion − remin
    # DOC: remainder of phyto mort + zoo excretion (already in NH4 pool),
    #      slow remineralisation back to DIC.
    remin_DOC_s = cfg.remin_DOC * day_s
    phyto_mort_tot = mpl_s * (Pd_ + Pn_)
    dPOC = (cfg.f_POC * phyto_mort_tot + mzq_s*Z_**2 + egestion
            - remin_s * POC_)      # POC remineralises at same rate as Det N
    dDOC = ((1.0 - cfg.f_POC) * phyto_mort_tot * cfg.R_CN
            - remin_DOC_s * DOC_)

    # ---- DIC ----
    # Net photosynthesis - remineralisation (N-cycle, Redfield C:N)
    dNO3_bio = dNO3 + dNH4    # total inorganic N change from biology
    dDIC = (-cfg.R_CN * mu_tot     # uptake
            + cfg.R_CN * (remin_s*D_ + remin_DON_s*DON_ + excretion)  # regeneration
            - remin_DOC_s * DOC_   # DOC → DIC
            )

    # ---- ALK ----
    # NO3 assimilation: +1 per mol; NH4 assimilation: -1 per mol; nitrif: -2 per mol
    dALK = mu_NO3 - mu_NH4 - 2.0 * nitrif

    return (dNO3, dNH4, dPO4, dSi, dFe,
            dPd, dPn, dChl_d, dChl_n,
            dZoo, dDet, dDON, dDOP,
            dDIC, dPOC, dDOC, dALK)
