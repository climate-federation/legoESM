"""Configuration and state containers for ocean biogeochemistry.

Two schemes:
- ``"abiotic"``: DIC + alkalinity only. Carbonate chemistry with air-sea
  CO2 exchange but no biology. Suitable for carbon-cycle experiments.
- ``"npzd"``: Nutrient-Phytoplankton-Zooplankton-Detritus ecosystem model
  coupled to the inorganic carbon cycle. Adds N, P, Z, D tracers with
  light-limited growth, grazing, mortality, and remineralization.
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

    NPZD parameters
    ---------------
    mu_max : float
        Maximum phytoplankton growth rate [1/day]. Default 1.5.
    k_N : float
        Nutrient half-saturation [mol N/m^3]. Default 0.7e-3.
    alpha_P : float
        Initial slope of P-I curve [1/(W/m^2)/day]. Default 0.025.
    g_max : float
        Maximum zooplankton grazing rate [1/day]. Default 0.6.
    k_P : float
        Grazing half-saturation [mol N/m^3]. Default 0.2e-3.
    gamma_Z : float
        Zooplankton assimilation efficiency. Default 0.7.
    m_P : float
        Phytoplankton linear mortality [1/day]. Default 0.05.
    m_Z : float
        Zooplankton quadratic mortality [1/(mol N/m^3)/day]. Default 0.2.
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

    # NPZD
    mu_max: float = 1.5
    k_N: float = 0.7e-3
    alpha_P: float = 0.025
    g_max: float = 0.6
    k_P: float = 0.2e-3
    gamma_Z: float = 0.7
    m_P: float = 0.05
    m_Z: float = 0.2
    remin_rate: float = 0.05
    w_sink: float = 10.0
    k_w_atten: float = 0.04
    k_chl_atten: float = 25.0
    R_CN: float = 6.625
    R_CaP: float = 0.07

    # Initial conditions
    DIC_init: float = 2.1
    ALK_init: float = 2.3
    NO3_init_surf: float = 5.0e-3
    NO3_init_deep: float = 30.0e-3
    Phyto_init: float = 0.1e-3
    Zoo_init: float = 0.05e-3
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
