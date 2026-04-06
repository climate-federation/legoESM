"""Configuration for the fully coupled Earth System Model.

Defines ``CoupledConfig`` which selects ocean, land, and carbon cycle
modes, and preset factories for standard configurations.
"""

from __future__ import annotations

from typing import NamedTuple

from legoesm.ocean.simple_ocean import SimpleOceanConfig
from legoesm.land.config import LandConfig, MultiLayerLandConfig
from legoesm.land.carbon.config import CarbonConfig
from legoesm.land.richards import RichardsConfig  # noqa: F401 (used in docstring)


class CoupledConfig(NamedTuple):
    """Configuration for the coupled ESM driver.

    Selects ocean mode, land complexity, and carbon cycle options.
    Each field maps to an existing component — no new physics is
    introduced; this is purely wiring configuration.

    Parameters
    ----------
    ocean_mode : str
        ``"slab"`` (default), ``"two_layer"``, or ``"dynamic"`` (3D ocean).
    ocean_config : SimpleOceanConfig
        Slab/two-layer ocean parameters (ignored when ``ocean_mode="dynamic"``).
    land_mode : str
        ``"none"`` (aquaplanet), ``"slab"`` (bucket), or ``"multilayer"``
        (Richards' equation soil hydrology).
    land_config : LandConfig or MultiLayerLandConfig
        Land parameters.  Type should match ``land_mode``:
        ``LandConfig`` for ``"slab"``, ``MultiLayerLandConfig`` for
        ``"multilayer"``.  Ignored when ``land_mode="none"``.
    use_pft : bool
        Use PFT-weighted land parameters instead of scalar defaults.
    carbon_active : bool
        Master switch for the carbon cycle.
    carbon_land : str
        Land carbon scheme: ``"none"``, ``"differland"`` (DALEC 6-pool),
        ``"seasonal"`` (prescribed NEE cycle).
    carbon_ocean : bool
        Enable ocean biogeochemistry CO2 exchange.
    co2_tracer : bool
        Enable prognostic atmospheric CO2 transport.
    co2_ppmv_init : float
        Initial atmospheric CO2 concentration [ppmv].
    f_land_mode : str
        ``"zero"`` (aquaplanet), ``"analytical"`` (default land mask),
        ``"file"`` (external land mask).
    coupling_dt : float
        Surface coupling interval [s].
    """
    # Ocean
    ocean_mode: str = "slab"
    ocean_config: SimpleOceanConfig = SimpleOceanConfig(mode="slab")
    # Land
    land_mode: str = "slab"
    land_config: LandConfig | MultiLayerLandConfig = LandConfig()
    use_pft: bool = False
    # Carbon cycle
    carbon_active: bool = False
    carbon_land: str = "none"
    carbon_ocean: bool = False
    co2_tracer: bool = False
    co2_ppmv_init: float = 415.0
    # Tile fractions
    f_land_mode: str = "analytical"
    # Coupling
    coupling_dt: float = 3600.0


# ============================================================================
# Preset factories
# ============================================================================

def preset_aquaplanet(**overrides) -> CoupledConfig:
    """Slab ocean everywhere, no land, no carbon."""
    defaults = dict(
        ocean_mode="slab",
        ocean_config=SimpleOceanConfig(mode="slab", h_mix=50.0),
        land_mode="none",
        f_land_mode="zero",
        carbon_active=False,
    )
    defaults.update(overrides)
    return CoupledConfig(**defaults)


def preset_slab_simple(**overrides) -> CoupledConfig:
    """Slab ocean + slab bucket land, no carbon."""
    defaults = dict(
        ocean_mode="slab",
        ocean_config=SimpleOceanConfig(mode="slab", h_mix=50.0),
        land_mode="slab",
        land_config=LandConfig(),
        f_land_mode="analytical",
        carbon_active=False,
    )
    defaults.update(overrides)
    return CoupledConfig(**defaults)


def preset_slab_pft(**overrides) -> CoupledConfig:
    """Slab ocean + slab land with PFT-weighted parameters."""
    defaults = dict(
        ocean_mode="slab",
        ocean_config=SimpleOceanConfig(mode="slab", h_mix=50.0),
        land_mode="slab",
        land_config=LandConfig(),
        use_pft=True,
        f_land_mode="analytical",
        carbon_active=False,
    )
    defaults.update(overrides)
    return CoupledConfig(**defaults)


def preset_slab_richards(**overrides) -> CoupledConfig:
    """Slab ocean + multi-layer Richards' equation land, no carbon."""
    defaults = dict(
        ocean_mode="slab",
        ocean_config=SimpleOceanConfig(mode="slab", h_mix=50.0),
        land_mode="multilayer",
        land_config=MultiLayerLandConfig(
            # Richards equation is always active in multilayer mode
        ),
        f_land_mode="analytical",
        carbon_active=False,
    )
    defaults.update(overrides)
    return CoupledConfig(**defaults)


def preset_slab_carbon(**overrides) -> CoupledConfig:
    """Slab ocean + Richards' land + DifferLand carbon + atm CO2 tracer."""
    defaults = dict(
        ocean_mode="slab",
        ocean_config=SimpleOceanConfig(mode="slab", h_mix=50.0),
        land_mode="multilayer",
        land_config=MultiLayerLandConfig(
            # Richards equation is always active in multilayer mode
            carbon=CarbonConfig(scheme="differland"),
        ),
        f_land_mode="analytical",
        carbon_active=True,
        carbon_land="differland",
        co2_tracer=True,
    )
    defaults.update(overrides)
    return CoupledConfig(**defaults)


def preset_full_coupled(**overrides) -> CoupledConfig:
    """Slab ocean + Richards' land + land & ocean carbon + atm CO2."""
    defaults = dict(
        ocean_mode="slab",
        ocean_config=SimpleOceanConfig(mode="slab", h_mix=50.0),
        land_mode="multilayer",
        land_config=MultiLayerLandConfig(
            # Richards equation is always active in multilayer mode
            carbon=CarbonConfig(scheme="differland"),
        ),
        f_land_mode="analytical",
        carbon_active=True,
        carbon_land="differland",
        carbon_ocean=True,
        co2_tracer=True,
    )
    defaults.update(overrides)
    return CoupledConfig(**defaults)


PRESETS = {
    "aquaplanet": preset_aquaplanet,
    "slab_simple": preset_slab_simple,
    "slab_pft": preset_slab_pft,
    "slab_richards": preset_slab_richards,
    "slab_carbon": preset_slab_carbon,
    "full_coupled": preset_full_coupled,
}
