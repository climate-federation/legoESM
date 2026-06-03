"""Land model configuration."""

from __future__ import annotations

from typing import Any, NamedTuple

from legoesm.land.carbon.config import CarbonConfig, StomataConfig
from legoesm.land.soil_grid import SoilGridConfig
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
from legoesm.land.soil_thermal import SoilThermalConfig
from legoesm.land.richards import RichardsConfig
from legoesm.land.surface_scheme import SimpleSEBConfig
from legoesm.surface_albedo import LandAlbedoConfig


class LandConfig(NamedTuple):
    """Slab land + bucket hydrology configuration."""
    C_soil: float = 2.0e6       # Soil heat capacity [J/m3/K]
    d_soil: float = 1.0         # Slab soil depth [m]
    W_max: float = 150.0        # Bucket capacity [kg/m2]
    albedo_land: float = 0.2    # Fallback constant albedo
    emissivity_land: float = 0.96
    z0_land: float = 0.05       # Roughness length [m]
    Cd_land: float = 3.0e-3     # Land drag coefficient (constant scheme)
    Ch_land: float = 3.0e-3     # Land heat transfer coefficient (constant)
    beta_min: float = 0.1       # Minimum moisture availability (dry soil)
    bulk_scheme: str = "constant"  # "constant" or "most"
    z_ref: float = 10.0           # Reference height for MOST [m]
    bulk_n_iter: int = 5          # MOST iterations
    # Snow/albedo
    snow_albedo_feedback: bool = False  # Enable snow albedo feedback
    land_albedo: LandAlbedoConfig = LandAlbedoConfig()
    T_snow_melt: float = 273.15  # Temperature above which snow melts [K]
    snow_melt_rate: float = 5.0e-6  # Snowmelt rate [kg/m2/s/K above T_melt]
    # Carbon cycle
    carbon: CarbonConfig = CarbonConfig()
    # Stomatal conductance / plant physiology
    stomata: StomataConfig = StomataConfig()
    # Surface scheme: ``SimpleSEBConfig`` (default) or ``TwoLeafCanopyConfig``.
    # Type hint is ``Any`` because NamedTuple does not support Unions well;
    # dispatch is done via ``isinstance`` inside ``step_land``.
    surface_scheme: Any = SimpleSEBConfig()


class MultiLayerLandConfig(NamedTuple):
    """Multi-layer soil model configuration (Task 8)."""
    # Surface properties
    albedo_land: float = 0.2    # Fallback constant albedo
    emissivity_land: float = 0.96
    z0_land: float = 0.05
    # Bulk flux
    Cd_land: float = 3.0e-3
    Ch_land: float = 3.0e-3
    beta_min: float = 0.1
    bulk_scheme: str = "constant"
    z_ref: float = 10.0
    bulk_n_iter: int = 5
    # Snow/albedo
    snow_albedo_feedback: bool = False  # Enable snow albedo feedback
    land_albedo: LandAlbedoConfig = LandAlbedoConfig()
    T_snow_melt: float = 273.15
    snow_melt_rate: float = 5.0e-6
    # Root water uptake
    root_depth: float = 1.0       # Root e-folding depth [m]
    theta_wp: float = 0.15        # Wilting point volumetric water content
    theta_fc: float = 0.30        # Field capacity volumetric water content
    # Sub-configs
    soil_grid: SoilGridConfig = SoilGridConfig()
    hydraulics: SoilHydraulicsConfig = SoilHydraulicsConfig()
    thermal: SoilThermalConfig = SoilThermalConfig()
    richards: RichardsConfig = RichardsConfig()
    # Carbon cycle
    carbon: CarbonConfig = CarbonConfig()
    # Stomatal conductance / plant physiology
    stomata: StomataConfig = StomataConfig()
    # Surface scheme: ``SimpleSEBConfig`` (default), ``TwoLeafCanopyConfig``,
    # or ``CLMMLCanopyConfig``.  Runtime dispatch via ``isinstance`` inside
    # ``step_multilayer_land``.
    surface_scheme: Any = SimpleSEBConfig()
