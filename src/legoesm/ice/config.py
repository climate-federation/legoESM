"""Sea ice model configuration."""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants
from legoesm.surface_albedo import IceAlbedoConfig


class SeaIceConfig(NamedTuple):
    """Thermodynamic slab + optional dynamics sea ice configuration.

    Dynamics modes:
    - ``"none"``: Slab thermodynamics only (diagnostic free-drift).
    - ``"free_drift"``: Free-drift velocity with tracer advection.
    - ``"evp"``: Elastic-Viscous-Plastic rheology (Hunke & Dukowicz 1997).

    Multi-category ice:
    - ``n_categories=1``: Single-category slab (default, backward compatible).
    - ``n_categories=5``: 5-category CICE-standard ITD.
    """
    rho_ice: float = constants.rho_ice
    c_ice: float = constants.c_pi
    k_ice: float = 2.04             # Ice thermal conductivity [W/m/K]
    L_f: float = constants.L_f
    h_ice_min: float = 0.01         # Min ice thickness for smooth ops [m]
    albedo_ice: float = 0.65        # Fallback constant albedo
    albedo_ocean: float = 0.06      # Ocean albedo for open-water freezing calc
    emissivity_ice: float = 0.97
    z0_ice: float = 5e-4            # Ice roughness length [m]
    Cd_ice: float = 1.5e-3          # Ice-atmosphere drag coefficient (constant)
    Ch_ice: float = 1.5e-3          # Ice-atmosphere heat transfer coeff (constant)
    # Transport
    drag_ocean: float = 5.5e-3      # Ocean-ice drag coefficient
    drag_atm: float = 1.3e-3        # Air-ice drag coefficient
    rho_air_ref: float = constants.rho_air
    rho_ocean_ref: float = constants.rho_ocean
    T_freeze_ocean: float = constants.T_freeze_ocean
    T_ice_min: float = 180.0        # Lower bound for numerical stability [K]
    ocean_heat_transfer_coeff: float = 20.0  # Ocean-ice heat transfer [W/m^2/K]
    # Concentration dynamics
    h_new_ice: float = 0.05         # Thickness for new ice formation [m]
    # Bulk flux algorithm
    bulk_scheme: str = "constant"   # "constant" or "most"
    z_ref: float = 10.0             # Reference height for MOST [m]
    bulk_n_iter: int = 5            # MOST iterations
    # Temperature-dependent albedo (Task 10)
    temp_dependent_albedo: bool = False
    ice_albedo: IceAlbedoConfig = IceAlbedoConfig()
    # --- Dynamics ---
    dynamics: str = "none"          # "none", "free_drift", or "evp"
    differentiable_dynamics: bool = False  # scan vs fori_loop for EVP
    # --- EVP rheology parameters ---
    N_evp: int = 120                # EVP subcycle count
    e_yield: float = 2.0            # Yield curve eccentricity
    P_star: float = 2.75e4          # Ice strength parameter [N/m^2]
    C_strength: float = 20.0        # Strength exponential decay constant
    Delta_min: float = 2.0e-9       # Minimum deformation rate [1/s]
    T_evp: float = 0.36             # EVP damping timescale ratio
    # --- Multi-category ice ---
    n_categories: int = 1           # 1=single-category (backward compat), 5=CICE ITD
    # --- Tracer transport ---
    transport: str = "none"         # "none" or "advect"
    # PPM transport is monotone only when |u·dt/dx| ≤ 1.  Default 1
    # substep covers the typical sea-ice regime (u ~ 0.1 m/s, dx ~
    # 50 km, dt ~ 1 h → C ~ 0.007).  Raise for storm / fine-grid
    # conditions where the Courant number can exceed 1.
    transport_subcycles: int = 1
