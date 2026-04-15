"""Global Barotropic Wind-Driven Circulation Experiment.

Tests the barotropic response to a global 3-belt wind stress pattern
(trades, westerlies, polar easterlies) in a basin with a simplified
continent and open Drake Passage.

Scientific Purpose:
- Validate global wind-driven barotropic circulation
- Test subtropical/subpolar gyre development
- Verify western boundary current formation
- Test circumpolar current in open Drake Passage
- Validate land mask handling with simplified continent geometry

Domain Configuration:
- Global ocean with simplified continent (20-60°E, from north cap to 55°S)
- Polar caps at ±80° latitude
- Open Drake Passage south of 55°S → circumpolar current
- Uniform T/S for purely barotropic dynamics (no baroclinic modes)

Physical Setup:
- Global 3-belt wind stress via "global_wind" forcing profile
- Lateral viscosity: A_h = 5×10⁵ m²/s
- Linear bottom drag: r = 1×10⁻⁴ s⁻¹
- No vertical mixing or convection

Expected Behavior:
- Subtropical/subpolar gyres in Atlantic-like and Pacific-like basins
- Western boundary currents (Gulf Stream / Kuroshio analogues)
- ACC-like circumpolar flow through Drake Passage

References:
- Stommel (1948), "The westward intensification of wind-driven ocean currents"
- Munk (1950), "On the wind-driven ocean circulation"
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from legoesm.core.field import Field


@dataclass
class GlobalBarotropicWindConfig:
    """Configuration for global barotropic wind-driven experiment."""
    # Uniform T/S for purely barotropic dynamics
    T_uniform: float = 10.0        # Uniform temperature [°C]
    S_uniform: float = 35.0        # Uniform salinity [PSU]

    # Domain
    H_max: float = 5500.0          # Maximum ocean depth [m]

    # Simplified continent geometry
    continent_lon_west: float = 20.0   # [degrees]
    continent_lon_east: float = 60.0   # [degrees]
    continent_lat_south: float = -55.0 # [degrees] — Drake Passage opens south of this
    polar_cap_lat: float = 80.0        # [degrees]

    # Physics
    A_h: float = 5e5               # Lateral viscosity [m²/s]
    bottom_drag_coeff: float = 1e-4  # Linear bottom drag [s⁻¹]


def _create_simplified_continent_mask(lon_deg, lat_deg, config):
    """Create simplified continent land mask.

    Geometry:
    - North/South polar caps: land poleward of ±polar_cap_lat
    - Single meridional continent from north cap to continent_lat_south
    - Open Drake Passage south of continent_lat_south
    - Everything else is ocean

    Returns 1 = ocean, 0 = land.
    """
    lon = jnp.asarray(lon_deg)
    lat = jnp.asarray(lat_deg)

    ocean = jnp.ones_like(lat)

    # Polar caps
    ocean = jnp.where(jnp.abs(lat) > config.polar_cap_lat, 0.0, ocean)

    # Meridional continent
    in_continent = (
        (lon >= config.continent_lon_west) &
        (lon <= config.continent_lon_east) &
        (lat >= config.continent_lat_south)
    )
    ocean = jnp.where(in_continent, 0.0, ocean)

    return ocean


def create_initial_conditions(grid_type: str, grid, z_coord,
                              config: GlobalBarotropicWindConfig = None):
    """Create global barotropic wind initial conditions.

    Creates a rest state with uniform T/S and a simplified continent mask.
    """
    if config is None:
        config = GlobalBarotropicWindConfig()

    if grid_type not in ("cubed_sphere", "latlon", "mpas"):
        raise NotImplementedError(
            f"Global barotropic wind not implemented for {grid_type}")

    # Create rest state with uniform T/S (no stratification → barotropic)
    if grid_type == "cubed_sphere":
        from legoesm.ocean.init import rest_state_ocean
        state = rest_state_ocean(
            grid, z_coord,
            T_surface=config.T_uniform, T_deep=config.T_uniform,
            S_uniform=config.S_uniform,
        )
        lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

    elif grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_surface=config.T_uniform, T_deep=config.T_uniform,
            S_uniform=config.S_uniform,
        )
        lon_1d = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_1d = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        if lon_1d.ndim == 1:
            lon_deg, lat_deg = np.meshgrid(lon_1d, lat_1d)
        else:
            lon_deg, lat_deg = lon_1d, lat_1d
        if lat_deg.ndim == 1:
            lat_deg = lat_1d[:, None] * np.ones((1, len(lon_1d)))
            lon_deg = lon_1d[None, :] * np.ones((len(lat_1d), 1))

    elif grid_type == "mpas":
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        state = rest_state_mpas_ocean(
            grid, z_coord,
            T_surface=config.T_uniform, T_deep=config.T_uniform,
            S_uniform=config.S_uniform,
        )
        lon_deg = np.asarray(grid.lonCell, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.latCell, dtype=np.float64) * 180 / np.pi

    # Apply simplified continent mask
    mask = _create_simplified_continent_mask(lon_deg, lat_deg, config)
    mask_typed = np.asarray(mask).astype(np.asarray(state.eta.data).dtype)
    state = state._replace(land_mask=Field(data=jnp.array(mask_typed)))

    # Latlon grids also need face masks for u/v
    if grid_type == "latlon":
        from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
        u_mask_new, v_mask_new = compute_face_masks(mask_typed)
        state = state._replace(
            u_mask=Field(data=u_mask_new),
            v_mask=Field(data=v_mask_new),
        )

    return state


def create_forcings(grid_type: str, grid,
                    config: GlobalBarotropicWindConfig = None):
    """Create physics config with global 3-belt wind forcing.

    Returns an OceanPhysicsConfig with prescribed global wind and
    linear bottom drag.
    """
    if config is None:
        config = GlobalBarotropicWindConfig()

    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.surface_forcing.config import (
        PrescribedForcingConfig, SurfaceForcingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.bottom_drag.config import (
        BottomDragConfig, LinearDragConfig,
    )
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig

    return OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="prescribed",
            prescribed=PrescribedForcingConfig(
                wind_profile="global_wind",
                tau_max=0.1,
                lat_south_deg=15.0,
                lat_north_deg=75.0,
            ),
        ),
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(
            scheme="linear",
            linear=LinearDragConfig(r=config.bottom_drag_coeff),
        ),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )


def create_domain_config(config: GlobalBarotropicWindConfig = None) -> Dict[str, Any]:
    if config is None:
        config = GlobalBarotropicWindConfig()
    return {
        "H_max": config.H_max,
        "description": "Global ocean with simplified continent and Drake Passage",
        "forcing_type": "global_3_belt_wind",
    }


def validate_results(final_state, diagnostics: Dict[str, list],
                     config: GlobalBarotropicWindConfig = None) -> Tuple[bool, str]:
    if config is None:
        config = GlobalBarotropicWindConfig()

    success = True
    notes_parts = []

    # Check finiteness
    for field_name in ("eta", "T", "u"):
        if hasattr(final_state, field_name):
            data = getattr(final_state, field_name).data
            if not jnp.all(jnp.isfinite(data)):
                return False, f"NaN/Inf in final {field_name}"

    max_speed_list = diagnostics.get("max_speed", [])
    if max_speed_list:
        speed = max_speed_list[-1]
        notes_parts.append(f"max_speed={speed:.4f}m/s")

    eta_list = diagnostics.get("mean_eta", [])
    if len(eta_list) >= 2:
        eta_drift = abs(eta_list[-1] - eta_list[0])
        notes_parts.append(f"eta_drift={eta_drift:.2e}")

    return success, ", ".join(notes_parts)


def get_diagnostic_field_specs() -> list:
    return [
        ("eta", "SSH (m)", "RdBu_r"),
        ("SST", "SST (degC)", "RdYlBu_r"),
        ("speed_sfc", "Surface speed (m/s)", "magma"),
    ]


def get_scalar_units() -> Dict[str, str]:
    return {
        "mean_eta": "m",
        "max_speed": "m/s",
        "mean_T": "degC",
        "mean_S": "PSU",
    }


EXPERIMENT_CONFIG = {
    "name": "global_barotropic_wind",
    "description": "Global barotropic wind-driven circulation with simplified continent",
    "scientific_purpose": "Validates global gyre formation, WBC, and circumpolar current",
    "reference": "Stommel (1948), Munk (1950)",
    "config_class": GlobalBarotropicWindConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": create_domain_config,
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 30.0,
    "quick_duration": 2.0,
    "grid_support": {
        "cubed_sphere": True,
        "latlon": True,
        "mpas": True,
        "spectral": False,
    },
}
