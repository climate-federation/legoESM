"""Sea ice model configuration."""

from __future__ import annotations

from typing import NamedTuple

from legoesm.surface_albedo import IceAlbedoConfig


class SeaIceConfig(NamedTuple):
    """Thermodynamic slab + simple transport sea ice configuration."""
    rho_ice: float = 917.0          # Ice density [kg/m3]
    c_ice: float = 2106.0           # Ice specific heat [J/kg/K]
    k_ice: float = 2.04             # Ice thermal conductivity [W/m/K]
    L_f: float = 3.337e5            # Latent heat of fusion [J/kg]
    h_ice_min: float = 0.01         # Min ice thickness for smooth ops [m]
    albedo_ice: float = 0.65        # Fallback constant albedo
    emissivity_ice: float = 0.97
    z0_ice: float = 5e-4            # Ice roughness length [m]
    Cd_ice: float = 1.5e-3          # Ice-atmosphere drag coefficient (constant)
    Ch_ice: float = 1.5e-3          # Ice-atmosphere heat transfer coeff (constant)
    # Transport
    drag_ocean: float = 5.5e-3      # Ocean-ice drag coefficient
    drag_atm: float = 1.3e-3        # Air-ice drag coefficient
    rho_air_ref: float = 1.225      # Reference air density [kg/m3]
    rho_ocean_ref: float = 1025.0   # Reference ocean density [kg/m3]
    T_freeze_ocean: float = 271.35  # Ocean freezing point [K]
    T_ice_min: float = 180.0        # Lower bound for numerical stability [K]
    # Concentration dynamics
    h_new_ice: float = 0.05         # Thickness for new ice formation [m]
    # Bulk flux algorithm
    bulk_scheme: str = "constant"   # "constant" or "most"
    z_ref: float = 10.0             # Reference height for MOST [m]
    bulk_n_iter: int = 5            # MOST iterations
    # Temperature-dependent albedo (Task 10)
    temp_dependent_albedo: bool = False
    ice_albedo: IceAlbedoConfig = IceAlbedoConfig()
