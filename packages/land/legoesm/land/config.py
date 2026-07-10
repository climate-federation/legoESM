"""Land model configuration."""

from __future__ import annotations

from typing import Any, NamedTuple

from legoesm import constants
from legoesm.land.carbon.config import CarbonConfig
from legoesm.land.stomata import StomataConfig
from legoesm.land.snow_bands import ElevationSnowBandConfig
from legoesm.land.soil_grid import SoilGridConfig
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
from legoesm.land.soil_thermal import SoilThermalConfig
from legoesm.land.topmodel_runoff import TopmodelConfig
from legoesm.land.richards import RichardsConfig
from legoesm.land.surface_scheme import SimpleSEBConfig
from legoesm.surface_albedo import LandAlbedoConfig


__param_spec__ = {
    "LandConfig": {
        "scheme_key": "land.slab",
        "excluded": {
            "beta_min": "numerics: soil-water stress floor",
            "z_ref": "convention: reference height [m]",
        },
        "params": {
            "C_soil": {"units": "1", "bounds": (660000.0, 6000000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "legoESM land surface", "shape": None},
            "Cd_land": {"units": "1", "bounds": (0.00099, 0.009), "tunable_tier": 2, "transform": "sigmoid", "category": "surface", "reference": "legoESM land surface", "shape": None},
            "Ch_land": {"units": "1", "bounds": (0.00099, 0.009), "tunable_tier": 2, "transform": "sigmoid", "category": "surface", "reference": "legoESM land surface", "shape": None},
            "W_max": {"units": "1", "bounds": (49.5, 450.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "legoESM land surface", "shape": None},
            "K_infiltration": {"units": "m/s", "bounds": (1e-7, 1e-4), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "Rawls Brakensiek Miller 1983 (Green-Ampt K_s)", "shape": None},
            "infil_suction_boost": {"units": "1", "bounds": (0.0, 10.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "Mein & Larson 1973 (Green-Ampt psi_f/L_f)", "shape": None},
            "albedo_land": {"units": "1", "bounds": (0.066, 0.6), "tunable_tier": 1, "transform": "sigmoid", "category": "radiation", "reference": "legoESM land surface", "shape": None},
            "d_soil": {"units": "1", "bounds": (0.33, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "legoESM land surface", "shape": None},
            "emissivity_land": {"units": "1", "bounds": (0.3168, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "radiation", "reference": "legoESM land surface", "shape": None},
            "snow_melt_rate": {"units": "1", "bounds": (1.65e-06, 1.5e-05), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "legoESM land surface", "shape": None},
            "z0_land": {"units": "1", "bounds": (0.0165, 0.15), "tunable_tier": 2, "transform": "sigmoid", "category": "surface", "reference": "legoESM land surface", "shape": None},
        },
    },
    "MultiLayerLandConfig": {
        "scheme_key": "land.multilayer",
        "excluded": {
            "beta_min": "numerics: soil-water stress floor",
            "z_ref": "convention: reference height [m]",
        },
        "params": {
            "Cd_land": {"units": "1", "bounds": (0.00099, 0.009), "tunable_tier": 2, "transform": "sigmoid", "category": "surface", "reference": "legoESM land surface", "shape": None},
            "Ch_land": {"units": "1", "bounds": (0.00099, 0.009), "tunable_tier": 2, "transform": "sigmoid", "category": "surface", "reference": "legoESM land surface", "shape": None},
            "albedo_land": {"units": "1", "bounds": (0.066, 0.6), "tunable_tier": 1, "transform": "sigmoid", "category": "radiation", "reference": "legoESM land surface", "shape": None},
            "emissivity_land": {"units": "1", "bounds": (0.3168, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "radiation", "reference": "legoESM land surface", "shape": None},
            "root_depth": {"units": "1", "bounds": (0.33, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "legoESM land surface", "shape": None},
            "snow_melt_rate": {"units": "1", "bounds": (1.65e-06, 1.5e-05), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "legoESM land surface", "shape": None},
            "soil_evap_resistance_exp": {"units": "1", "bounds": (0.0, 6.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "Sellers 1992 / Lee & Pielke 1992 (bare-soil evap resistance)", "shape": None},
            "soil_evap_litter_resistance_s_m": {"units": "s m-1", "bounds": (0.0, 400.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "Sakaguchi & Zeng 2009 (forest-floor litter resistance)", "shape": None},
            "theta_fc": {"units": "1", "bounds": (0.099, 0.9), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "legoESM land surface", "shape": None},
            "theta_wp": {"units": "1", "bounds": (0.0495, 0.45), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "legoESM land surface", "shape": None},
            "z0_land": {"units": "1", "bounds": (0.0165, 0.15), "tunable_tier": 2, "transform": "sigmoid", "category": "surface", "reference": "legoESM land surface", "shape": None},
        },
    },
}


class LandConfig(NamedTuple):
    """Slab land + bucket hydrology configuration."""
    C_soil: float = 2.0e6       # Soil heat capacity [J/m3/K]
    d_soil: float = 1.0         # Slab soil depth [m]
    W_max: float = 150.0        # Bucket capacity [kg/m2]
    # Bucket runoff partition (Green-Ampt infiltration excess + saturation excess)
    K_infiltration: float = 1.0e-5     # Saturated infiltration capacity K_s [m/s]
    infil_suction_boost: float = 2.0   # Green-Ampt suction enhancement psi_f/L_f [-]
    infiltration_excess: bool = True   # Enable Hortonian infiltration-excess runoff
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
    T_snow_melt: float = constants.T_freeze
    snow_melt_rate: float = 5.0e-6  # Snowmelt rate [kg/m2/s/K above T_melt]
    # Carbon cycle
    carbon: CarbonConfig = CarbonConfig()
    # Stomatal conductance / plant physiology
    stomata: StomataConfig = StomataConfig()
    # Surface scheme: ``SimpleSEBConfig`` (default) or ``TwoLeafCanopyConfig``.
    # Type hint is ``Any`` because NamedTuple does not support Unions well;
    # dispatch is done via ``isinstance`` inside ``step_land``.
    surface_scheme: Any = SimpleSEBConfig()
    # Runoff scheme (appended for positional-ABI stability): "bucket" (default,
    # Green-Ampt Hortonian + Dunne saturation-excess, byte-identical) or
    # "topmodel" (SIMTOP sub-grid saturated fraction + topographic baseflow,
    # Niu 2005 / CLM4.5).  Unknown -> ValueError at dispatch.
    runoff_scheme: str = "bucket"
    topmodel: TopmodelConfig = TopmodelConfig()


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
    # Surface soil resistance to BARE-SOIL evaporation.  The top soil layer dries
    # into a high-resistance crust far faster than the root-zone mean, so bare-soil
    # evaporation is throttled by the TOP-layer effective saturation S_top via a
    # beta-method efficiency S_top**soil_evap_resistance_exp (Sellers 1992 / Lee &
    # Pielke 1992 magnitude at exp=2).  exp=0 disables it (legacy: bare-soil evap
    # limited only by the root-zone beta + whole-column water supply -> over-strong
    # soil evaporation and a too-fast surface dry-down).
    soil_evap_resistance_exp: float = 2.0
    # Below-canopy soil-evaporation SERIES-RESISTANCE formulation (canopy path).
    # When True (default), the top-layer moisture control on soil evaporation is a
    # Sellers-1992 surface resistance r_ss = exp(8.206 - 4.255 theta_1/theta_sat)
    # PLUS a Sakaguchi-Zeng-2009 forest-floor litter resistance, added in SERIES
    # with the below-canopy aerodynamic resistance in the two-leaf soil energy
    # balance (energy_balance.soil_surface_evap_resistance); the beta efficiency
    # S_top**soil_evap_resistance_exp is then bypassed (mutually exclusive — same
    # Sellers-1992 physics).  The aerodynamic-only path (r_ss omitted) let a wet
    # forest floor evaporate at near-potential rate; the resistance form throttles
    # a WET surface and self-scales with canopy density across biomes.  Static
    # Python bool (feature gate).  False -> legacy beta efficiency (SimpleSEB path
    # is unaffected either way).
    soil_evap_series_resistance: bool = True
    # Reference forest-floor litter resistance [s/m] (Sakaguchi & Zeng 2009),
    # scaled by litter cover 1 - exp(-0.5 LAI); ~0 for bare soil, saturating for a
    # closed canopy.  Only used when soil_evap_series_resistance is True.  Default
    # 300 s/m calibrated on the FLUXNET EC-site validation (4 biomes: US-MMS DBF,
    # DE-Obe ENF, US-Ton savanna, US-Var grassland) to give a physical closed-
    # canopy soil-evaporation fraction (JJA ~0.2 forest, ~0 Mediterranean-summer)
    # and unbiased LE vs energy-closure-corrected obs; well within the published
    # forest-floor range (~1e2-1e3 s/m).
    soil_evap_litter_resistance_s_m: float = 300.0
    bulk_scheme: str = "constant"
    z_ref: float = 10.0
    bulk_n_iter: int = 5
    # Snow/albedo
    snow_albedo_feedback: bool = False  # Enable snow albedo feedback
    land_albedo: LandAlbedoConfig = LandAlbedoConfig()
    T_snow_melt: float = constants.T_freeze
    snow_melt_rate: float = 5.0e-6
    # Sub-grid elevation-band snow (VIC snow bands / CESM MEC); ``None`` (default)
    # runs the single cell-mean snowpack.  See ``legoesm.land.snow_bands``.
    elev_bands: ElevationSnowBandConfig | None = None
    # Root water uptake
    root_depth: float = 1.0       # Root e-folding depth [m]
    theta_wp: float = 0.15        # Wilting point volumetric water content
    theta_fc: float = 0.30        # Field capacity volumetric water content
    # Sub-configs
    soil_grid: SoilGridConfig = SoilGridConfig()
    hydraulics: SoilHydraulicsConfig = SoilHydraulicsConfig()
    thermal: SoilThermalConfig = SoilThermalConfig()
    richards: RichardsConfig = RichardsConfig(fc_drain_saturation=0.5)
    # Carbon cycle
    carbon: CarbonConfig = CarbonConfig()
    # Stomatal conductance / plant physiology
    stomata: StomataConfig = StomataConfig()
    # Surface scheme: ``SimpleSEBConfig`` (default), ``TwoLeafCanopyConfig``,
    # or ``CLMMLCanopyConfig``.  Runtime dispatch via ``isinstance`` inside
    # ``step_multilayer_land``.
    surface_scheme: Any = SimpleSEBConfig()


def resolve_land_config(land_mode: str, land_config=None):
    """Return the land config object matching ``land_mode``.

    Single source of truth for the ``land_mode`` -> config-type mapping used by
    the coupled driver and the ``run_lmip_smoke`` driver: ``"multilayer"`` ->
    :class:`MultiLayerLandConfig`, ``"slab"``/``"none"`` -> :class:`LandConfig`.
    A ``land_config`` of the wrong type for the mode is replaced with the
    mode's default (so the runtime type always matches the selected model).
    """
    if land_mode == "multilayer":
        return land_config if isinstance(land_config, MultiLayerLandConfig) else MultiLayerLandConfig()
    if land_mode == "none":
        return LandConfig()
    return land_config if isinstance(land_config, LandConfig) else LandConfig()
