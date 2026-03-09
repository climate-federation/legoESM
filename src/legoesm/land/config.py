"""Land model configuration."""

from __future__ import annotations

from typing import NamedTuple

from legoesm.land.soil_grid import SoilGridConfig
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
from legoesm.land.soil_thermal import SoilThermalConfig
from legoesm.land.richards import RichardsConfig


class LandConfig(NamedTuple):
    """Slab land + bucket hydrology configuration."""
    C_soil: float = 2.0e6       # Soil heat capacity [J/m3/K]
    d_soil: float = 1.0         # Slab soil depth [m]
    W_max: float = 150.0        # Bucket capacity [kg/m2]
    albedo_land: float = 0.2
    emissivity_land: float = 0.96
    z0_land: float = 0.05       # Roughness length [m]
    Cd_land: float = 3.0e-3     # Land drag coefficient (constant scheme)
    Ch_land: float = 3.0e-3     # Land heat transfer coefficient (constant)
    beta_min: float = 0.1       # Minimum moisture availability (dry soil)
    bulk_scheme: str = "constant"  # "constant" or "most"
    z_ref: float = 10.0           # Reference height for MOST [m]
    bulk_n_iter: int = 5          # MOST iterations


class MultiLayerLandConfig(NamedTuple):
    """Multi-layer soil model configuration (Task 8)."""
    # Surface properties
    albedo_land: float = 0.2
    emissivity_land: float = 0.96
    z0_land: float = 0.05
    # Bulk flux
    Cd_land: float = 3.0e-3
    Ch_land: float = 3.0e-3
    beta_min: float = 0.1
    bulk_scheme: str = "constant"
    z_ref: float = 10.0
    bulk_n_iter: int = 5
    # Sub-configs
    soil_grid: SoilGridConfig = SoilGridConfig()
    hydraulics: SoilHydraulicsConfig = SoilHydraulicsConfig()
    thermal: SoilThermalConfig = SoilThermalConfig()
    richards: RichardsConfig = RichardsConfig()
