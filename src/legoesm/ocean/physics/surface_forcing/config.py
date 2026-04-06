"""Configuration for ocean surface forcing schemes."""

from __future__ import annotations

from typing import NamedTuple


class PrescribedForcingConfig(NamedTuple):
    """Fixed wind stress and heat/freshwater fluxes."""
    tau_x: float = 0.0           # Zonal wind stress [N/m^2]
    tau_y: float = 0.0           # Meridional wind stress [N/m^2]
    Q_net: float = 0.0           # Net surface heat flux [W/m^2] (+ into ocean)
    E_minus_P: float = 0.0       # Evaporation minus precipitation [m/s]
    wind_profile: str = "constant"   # "constant", "cosine_latitude", "single_gyre", or "double_gyre"
    tau_max: float = 0.1         # Max wind stress for wind profiles [N/m^2]
    lat_south_deg: float = 15.0  # Southern basin boundary [degrees]
    lat_north_deg: float = 75.0  # Northern basin boundary [degrees]


class RestoringConfig(NamedTuple):
    """SST/SSS restoring to target profiles."""
    tau_T: float = 2592000.0     # Temperature restoring timescale [s] (30 days)
    tau_S: float = 2592000.0     # Salinity restoring timescale [s] (30 days)
    T_star_eq: float = 25.0      # Target equatorial SST [degC]
    T_star_pole: float = 0.0     # Target polar SST [degC]
    S_star: float = 35.0         # Target SSS [PSU]
    T_profile: str = "cosine"    # "cosine" or "constant"


class BulkFormulaConfig(NamedTuple):
    """COARE-like air-sea flux formulation."""
    C_D: float = 1.5e-3     # Drag coefficient (constant scheme)
    C_H: float = 1.5e-3     # Sensible heat transfer coefficient (constant)
    C_E: float = 1.5e-3     # Latent heat transfer coefficient (constant)
    rho_a: float = 1.225    # Air density [kg/m^3]
    c_pa: float = 1004.64   # Specific heat of air [J/(kg·K)] (= constants.c_pd)
    L_v: float = 2.501e6    # Latent heat of vaporization [J/kg] (= constants.L_v)
    T_a: float = 280.0      # Air temperature [K]
    U_a: float = 5.0        # Wind speed [m/s]
    q_a: float = 0.005      # Air specific humidity [kg/kg]
    SW_down: float = 200.0  # Downward shortwave [W/m^2]
    LW_down: float = 300.0  # Downward longwave [W/m^2]
    bulk_scheme: str = "constant"  # "constant", "coare3", "large_yeager"
    z_ref: float = 10.0     # Reference height for MOST [m]
    z0: float = 1e-4        # Roughness length for MOST [m]
    bulk_n_iter: int = 5    # MOST iterations


class SurfaceForcingConfig(NamedTuple):
    """Top-level surface forcing configuration."""
    scheme: str = "none"  # "prescribed", "restoring", "bulk_formulas", "none"
    prescribed: PrescribedForcingConfig = PrescribedForcingConfig()
    restoring: RestoringConfig = RestoringConfig()
    bulk_formulas: BulkFormulaConfig = BulkFormulaConfig()
