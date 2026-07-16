"""Configuration and state containers for ocean biogeochemistry.

Two schemes:
- ``"abiotic"``: DIC + alkalinity only. Carbonate chemistry with air-sea
  CO2 exchange but no biology. Suitable for carbon-cycle experiments.
- ``"npzd"``: Nutrient-Phytoplankton-Zooplankton-Detritus ecosystem model
  coupled to the inorganic carbon cycle. Adds N, P, Z, D tracers with
  light-limited growth, grazing, mortality, and remineralization.
- ``"npzd_v2"``: Extended 16-tracer scheme adding two phytoplankton
  functional types (diatoms + nanophyto each with prognostic Chl), NH4
  cycle with nitrification, DOP, POC, DOC, and ALK.
- ``"none"``: Disabled (default).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class BiogeoConfig(NamedTuple):
    """Ocean biogeochemistry configuration.

    Parameters
    ----------
    scheme : str
        ``"none"``, ``"abiotic"``, or ``"npzd"``.

    Carbonate chemistry (abiotic + npzd)
    -------------------------------------
    pCO2_atm : float
        Atmospheric CO2 partial pressure [uatm]. Default 400.
    wind_speed : float
        Constant 10-m wind speed [m/s] for gas exchange (used when
        no atmospheric coupling provides wind). Default 7.0.
    K_h_bio : float
        Horizontal diffusivity for biogeo tracers [m^2/s]. Default 1e3.
    K_v_bio : float
        Vertical diffusivity for biogeo tracers [m^2/s]. Default 1e-4.

    NPZD parameters
    ---------------
    mu_max : float
        Maximum phytoplankton growth rate [1/day]. Default 1.5.
    k_N : float
        Nutrient half-saturation [mol N/m^3]. Default 0.7e-3.
    k_PAR : float
        Light half-saturation [W/m^2]. Default 30.0.
    alpha_P : float
        Initial slope of P-I curve [1/(W/m^2)/day]. Default 0.025.
    g_max : float
        Maximum zooplankton grazing rate [1/day]. Default 0.6.
    k_P : float
        Grazing half-saturation [mol N/m^3]. Default 0.2e-3.
    gamma_Z : float
        Zooplankton assimilation efficiency. Default 0.7.
    epse : float
        Zooplankton egestion efficiency (fraction of non-assimilated
        grazing routed to detritus rather than N). Default 0.5.
    m_P : float
        Phytoplankton linear mortality [1/day]. Default 0.05.
    m_Pq : float
        Phytoplankton quadratic aggregation/loss [m^3/(mol N)/day]. Default 0.2.
    m_Zl : float
        Zooplankton linear mortality [1/day]. Default 0.02.
    m_Z : float
        Zooplankton quadratic mortality [m^3/(mol N)/day]. Default 0.2.
    remin_rate : float
        Detritus remineralization rate [1/day]. Default 0.05.
    w_sink : float
        Detritus sinking speed [m/day]. Default 10.0.
    k_w_atten : float
        Seawater light attenuation coefficient [1/m]. Default 0.04.
    k_chl_atten : float
        Chlorophyll self-shading [m^2/(mol N)]. Default 25.0.
    R_CN : float
        Redfield C:N ratio [mol C / mol N]. Default 6.625.
    R_ON : float
        Redfield O2:N ratio [mol O2 / mol N]. Default 10.625.
    R_CaP : float
        Rain ratio (CaCO3 production / organic C export). Default 0.07.

    Initial conditions
    ------------------
    DIC_init : float
        Initial DIC [mol C/m^3]. Default 2.1 (~ 2100 umol/kg).
    ALK_init : float
        Initial alkalinity [mol eq/m^3]. Default 2.3 (~ 2300 umol/kg).
    NO3_init_surf : float
        Initial surface NO3 [mol N/m^3]. Default 5e-3.
    NO3_init_deep : float
        Initial deep NO3 [mol N/m^3]. Default 30e-3.
    Phyto_init : float
        Initial phytoplankton [mol N/m^3]. Default 0.1e-3.
    Zoo_init : float
        Initial zooplankton [mol N/m^3]. Default 0.05e-3.
    Det_init : float
        Initial detritus [mol N/m^3]. Default 0.01e-3.
    """
    scheme: str = "none"

    # Carbonate chemistry
    pCO2_atm: float = 400.0
    wind_speed: float = 7.0
    K_h_bio: float = 1.0e3
    K_v_bio: float = 1.0e-4

    # NPZD
    mu_max: float = 1.5
    k_N: float = 0.7e-3
    k_PAR: float = 30.0
    alpha_P: float = 0.025
    g_max: float = 0.6
    k_P: float = 0.2e-3
    gamma_Z: float = 0.7
    epse: float = 0.5
    m_P: float = 0.05
    m_Pq: float = 0.2
    m_Zl: float = 0.02
    m_Z: float = 0.2
    remin_rate: float = 0.05
    w_sink: float = 10.0
    k_w_atten: float = 0.04
    k_chl_atten: float = 25.0
    R_CN: float = 6.625
    R_ON: float = 10.625
    R_CaP: float = 0.07
    R_NP: float = 16.0          # Redfield N:P ratio [mol N / mol P]
    # Nutrient nudging toward WOA climatology
    nudge_nutrients: bool = False  # enable nutrient nudging
    tau_nudge_days: float = 365.0  # nudging timescale [days]
    NO3_target: object = None      # WOA NO3 target [mol/m3]
    PO4_target: object = None      # WOA PO4 target [mol/m3]
    Si_target:  object = None      # WOA Si  target [mol/m3]

    # ------------------------------------------------------------------ #
    # npzd_v2 — extended 16-tracer parameters                            #
    # ------------------------------------------------------------------ #

    # Diatoms (PFT 1)
    mu_max_d: float = 0.25        # max growth rate [1/day]
    k_N_d: float = 1.0e-4        # NO3+NH4 half-saturation [mol N/m^3]
    k_NH4_d: float = 5.0e-5      # NH4 half-saturation [mol N/m^3]
    k_Fe_d: float = 80.0e-12     # Fe half-saturation [mol Fe/m^3]  (80 nmol/m3 in SI)
    k_Si: float = 0.3e-3         # Si half-saturation [mol Si/m^3]
    R_SiN: float = 1.0           # Si:N uptake ratio [mol Si / mol N]
    R_FeN_d: float = 7.5e-6      # Fe:N for diatoms [mol Fe / mol N]  (Sunda & Huntsman 1995)
    theta_min_d: float = 0.5e-3  # min Chl:N [kg Chl / mol N]
    theta_max_d: float = 5.0e-3  # max Chl:N [kg Chl / mol N]

    # Nanophyto (PFT 2)
    mu_max_n: float = 0.20        # max growth rate [1/day]
    k_N_n: float = 1.0e-4        # NO3+NH4 half-saturation [mol N/m^3]
    k_NH4_n: float = 5.0e-5      # NH4 half-saturation [mol N/m^3]
    k_Fe_n: float = 50.0e-12     # Fe half-saturation [mol Fe/m^3]
    R_FeN_n: float = 5.0e-6      # Fe:N for nanophyto [mol Fe / mol N]
    theta_min_n: float = 0.5e-3  # min Chl:N [kg Chl / mol N]
    theta_max_n: float = 4.0e-3  # max Chl:N [kg Chl / mol N]

    # Shared phyto
    m_Pl: float = 0.15           # linear mortality [1/day]
    m_Pq_v2: float = 1.0         # quadratic aggregation [m^3/(mol N)/day] — increased to damp initial bloom
    psi_coeff: float = 1.5e3     # NH4 inhibition coefficient [m^3/mol N]

    # Nitrification
    k_nitrif: float = 0.20       # max nitrification rate [1/day]
    kI_nitrif: float = 0.005     # light inhibition [(W/m^2)^-1]

    # Zooplankton v2
    m_Zl_v2: float = 0.10        # linear mortality [1/day]

    # Detritus v2
    remin_opal: float = 0.03     # opal dissolution rate [1/day]
    remin_DON: float = 0.01      # DON remineralisation rate [1/day]

    # Iron
    k_Fe_scav: float = 5.0       # scavenging rate [1/day]  — aggressive to equilibrate in 5yr
    Fe_ligand: float = 150.0e-12 # scavenging threshold [mol Fe/m^3]
    Fe_dust: float = 5.0e-9      # aeolian deposition [mol Fe/m^2/day]
    R_FeN_avg: float = 6.0e-6    # average Fe:N for regeneration [mol/mol]

    # POC / DOC split
    f_POC: float = 0.7           # fraction of phyto mort → POC (rest → DOC)
    remin_DOC: float = 0.005     # DOC remineralisation rate [1/day]

    # Initial conditions v2
    NH4_init: float = 1.0e-4     # [mol N/m^3]
    Fe_init_surf: float = 50.0e-12   # [mol Fe/m^3]
    Fe_init_deep: float = 200.0e-12  # [mol Fe/m^3]
    Si_init_surf: float = 1.0e-3     # [mol Si/m^3]
    Si_init_deep: float = 15.0e-3    # [mol Si/m^3]

    # Initial conditions
    DIC_init: float = 2.1
    ALK_init: float = 2.3
    NO3_init_surf: float = 5.0e-3
    NO3_init_deep: float = 30.0e-3
    Phyto_init: float = 0.1e-3
    Zoo_init: float = 0.5e-3     # higher init to damp initial bloom
    Det_init: float = 0.01e-3


class OceanBiogeoState(NamedTuple):
    """Ocean biogeochemistry tracer state.

    All fields have shape matching the ocean grid:
    - Cubed-sphere: (6, n, n, nlev)
    - MPAS: (nCells, nlev)

    Fields are None when the corresponding scheme does not use them.
    """
    DIC: jax.Array          # Dissolved inorganic carbon [mol C/m^3]
    ALK: jax.Array          # Total alkalinity [mol eq/m^3]
    NO3: jax.Array | None = None  # Nitrate [mol N/m^3], NPZD only
    Phyto: jax.Array | None = None  # Phytoplankton [mol N/m^3], NPZD only
    Zoo: jax.Array | None = None    # Zooplankton [mol N/m^3], NPZD only
    Det: jax.Array | None = None    # Detritus [mol N/m^3], NPZD only
    # npzd_v2 additional tracers
    NH4: jax.Array | None = None       # Ammonium [mol N/m^3]
    PO4: jax.Array | None = None       # Phosphate [mol P/m^3]
    Si: jax.Array | None = None        # Silicate [mol Si/m^3]
    Fe: jax.Array | None = None        # Dissolved iron [mol Fe/m^3]
    Pd: jax.Array | None = None        # Diatom biomass N [mol N/m^3]
    Pn: jax.Array | None = None        # Nanophyto biomass N [mol N/m^3]
    Chl_d: jax.Array | None = None     # Diatom Chl [kg Chl/m^3]
    Chl_n: jax.Array | None = None     # Nanophyto Chl [kg Chl/m^3]
    DON: jax.Array | None = None       # Dissolved organic N [mol N/m^3]
    DOP: jax.Array | None = None       # Dissolved organic P [mol P/m^3]
    POC: jax.Array | None = None       # Particulate organic C [mol C/m^3]
    DOC: jax.Array | None = None       # Dissolved organic C [mol C/m^3]


class BiogeoTendencies(NamedTuple):
    """Source/sink tendencies for biogeochemistry tracers.

    Same shapes as OceanBiogeoState. These are the
    biogeochemistry-only tendencies (gas exchange, biology, etc.).
    Advection/diffusion tendencies are handled by the ocean dynamics.
    """
    dDIC_dt: jax.Array
    dALK_dt: jax.Array
    dNO3_dt: jax.Array | None = None
    dPhyto_dt: jax.Array | None = None
    dZoo_dt: jax.Array | None = None
    dDet_dt: jax.Array | None = None
    # npzd_v2
    dNH4_dt: jax.Array | None = None
    dPO4_dt: jax.Array | None = None
    dSi_dt: jax.Array | None = None
    dFe_dt: jax.Array | None = None
    dPd_dt: jax.Array | None = None
    dPn_dt: jax.Array | None = None
    dChl_d_dt: jax.Array | None = None
    dChl_n_dt: jax.Array | None = None
    dDON_dt: jax.Array | None = None
    dDOP_dt: jax.Array | None = None
    dPOC_dt: jax.Array | None = None
    dDOC_dt: jax.Array | None = None


class AirSeaCO2Diagnostics(NamedTuple):
    """Diagnostics from the air-sea CO2 flux calculation."""
    pCO2_ocean: jax.Array   # Ocean surface pCO2 [uatm]
    pH: jax.Array           # Surface pH
    flux_co2: jax.Array     # Air-sea CO2 flux [mol C/m^2/s], positive into ocean
    k_w: jax.Array          # Piston velocity [m/s]


def init_biogeo_state(
    shape_3d: tuple,
    z_full_ref: jax.Array,
    cfg: BiogeoConfig,
) -> OceanBiogeoState | None:
    """Initialize biogeochemistry state.

    Parameters
    ----------
    shape_3d : tuple
        Shape of 3D ocean arrays, e.g. (6, n, n, nlev) or (nCells, nlev).
    z_full_ref : jax.Array
        Reference depths [m], shape (nlev,), negative values.
    cfg : BiogeoConfig
        Configuration.

    Returns
    -------
    OceanBiogeoState or None
        None if scheme is "none".
    """
    if cfg.scheme == "none":
        return None

    nlev = shape_3d[-1]

    DIC = jnp.full(shape_3d, cfg.DIC_init)
    ALK = jnp.full(shape_3d, cfg.ALK_init)

    if cfg.scheme == "abiotic":
        return OceanBiogeoState(DIC=DIC, ALK=ALK)

    elif cfg.scheme == "npzd":
        # NO3: linear increase with depth from surface to deep
        # z_full_ref is negative, so deeper = more negative
        z_norm = jnp.clip(-z_full_ref / 1000.0, 0.0, 1.0)  # 0 at surface, 1 at 1000m+
        NO3_profile = cfg.NO3_init_surf + (cfg.NO3_init_deep - cfg.NO3_init_surf) * z_norm
        NO3 = jnp.broadcast_to(
            NO3_profile.reshape((1,) * (len(shape_3d) - 1) + (nlev,)),
            shape_3d,
        )

        Phyto = jnp.full(shape_3d, cfg.Phyto_init)
        Zoo = jnp.full(shape_3d, cfg.Zoo_init)
        Det = jnp.full(shape_3d, cfg.Det_init)

        return OceanBiogeoState(
            DIC=DIC, ALK=ALK, NO3=NO3,
            Phyto=Phyto, Zoo=Zoo, Det=Det,
        )

    raise ValueError(f"Unknown biogeochemistry scheme: {cfg.scheme!r}")

# ---- npzd_v2 init (appended) ----
# Monkey-patch init_biogeo_state to handle npzd_v2.
# This avoids editing the function body directly.
_orig_init_biogeo_state = init_biogeo_state

def init_biogeo_state(shape_3d, z_full_ref, cfg):
    if cfg.scheme != "npzd_v2":
        return _orig_init_biogeo_state(shape_3d, z_full_ref, cfg)

    import jax.numpy as _jnp
    nlev = shape_3d[-1]
    z_norm = _jnp.clip(-z_full_ref / 1000.0, 0.0, 1.0)
    _bc = lambda p: _jnp.broadcast_to(
        p.reshape((1,) * (len(shape_3d) - 1) + (nlev,)), shape_3d)

    NO3_p  = cfg.NO3_init_surf + (cfg.NO3_init_deep - cfg.NO3_init_surf) * z_norm
    Fe_p   = cfg.Fe_init_surf  + (cfg.Fe_init_deep  - cfg.Fe_init_surf)  * z_norm
    Si_p   = cfg.Si_init_surf  + (cfg.Si_init_deep  - cfg.Si_init_surf)  * z_norm

    return OceanBiogeoState(
        DIC    = _jnp.full(shape_3d, cfg.DIC_init),
        ALK    = _jnp.full(shape_3d, cfg.ALK_init),
        NO3    = _bc(NO3_p),
        NH4    = _jnp.full(shape_3d, cfg.NH4_init),
        PO4    = _bc(NO3_p / cfg.R_NP),   # Redfield from NO3
        Si     = _bc(Si_p),
        Fe     = _bc(Fe_p),
        Pd     = _jnp.full(shape_3d, cfg.Phyto_init * 0.5),
        Pn     = _jnp.full(shape_3d, cfg.Phyto_init * 0.5),
        Chl_d  = _jnp.full(shape_3d, cfg.Phyto_init * 0.5 * cfg.theta_min_d),
        Chl_n  = _jnp.full(shape_3d, cfg.Phyto_init * 0.5 * cfg.theta_min_n),
        Zoo    = _jnp.full(shape_3d, cfg.Zoo_init),
        Det    = _jnp.full(shape_3d, cfg.Det_init),
        DON    = _jnp.full(shape_3d, 1.0e-4),
        DOP    = _jnp.full(shape_3d, 1.0e-4 / cfg.R_NP),
        POC    = _jnp.full(shape_3d, cfg.Det_init * cfg.R_CN),
        DOC    = _jnp.full(shape_3d, 1.0e-3),
    )
