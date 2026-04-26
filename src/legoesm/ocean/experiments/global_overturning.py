"""Global Baroclinic Overturning Circulation Experiment.

Inspired by Wolfe & Cessi (2010, JPO) but adapted to spherical geometry
and coarse resolution (1-0.5 deg) as a stepping stone toward OMIP.

Scientific Purpose:
- Validate global baroclinic ocean dynamics with stratification
- Test meridional overturning circulation (MOC) development
- Verify Southern Ocean circumpolar current through Drake Passage
- Test interaction of wind-driven gyres with thermal stratification
- Benchmark surface temperature restoring in global domain
- Compare lat-lon and MPAS grids side-by-side

Domain Configuration:
- Global ocean with simplified continent (20-60 deg E, from north cap to 55 deg S)
- Open Drake Passage south of 55 deg S -> circumpolar current
- Polar caps at +/- 80 deg latitude
- Flat bottom at H_max depth
- Exponential temperature stratification (warm surface, cold abyss)

Physical Setup:
- Global 3-belt zonal wind stress (trades, westerlies, polar easterlies)
- Surface temperature restoring: cosine(lat) profile, warm equator -> cold poles
- Linear EOS: buoyancy depends on temperature only (beta_S = 0)
- Constant vertical viscosity and diffusivity
- Convective adjustment via enhanced diffusion where N^2 < 0
- Linear bottom drag
- 20 z-star levels with surface-intensified stretching

Expected Behavior:
- Subtropical/subpolar gyres in Atlantic-like and Pacific-like basins
- Western boundary currents (Gulf Stream / Kuroshio analogues)
- ACC-like circumpolar flow through Drake Passage
- Thermocline maintained by wind-driven overturning + restoring
- Deep stratification set by Southern Ocean channel dynamics

At 1 deg resolution eddies are NOT resolved.  Stratification and MOC
are maintained diffusively rather than by eddy fluxes.  This is
acceptable as a stepping stone; OMIP models also run at this resolution
(typically with GM/Redi parameterization, which is future work).

References:
- Wolfe & Cessi (2010), "What Sets the Strength of the Middepth
  Stratification and Overturning in Eddying Ocean Models?", JPO 40.
- Stommel (1948), "The westward intensification of wind-driven currents"
- Nikurashin & Vallis (2012) — global_wind profile shape
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from legoesm.core.field import Field
from legoesm.ocean.eos import LinearEOSConfig


@dataclass
class GlobalOverturningConfig:
    """Configuration for the global overturning circulation experiment."""

    # --- Stratification ---
    T_surface: float = 20.0        # Surface temperature [degC]
    T_deep: float = 2.0            # Abyssal temperature [degC]
    T_scale_depth: float = 1000.0  # Stratification e-folding depth [m]
    S_uniform: float = 35.0        # Uniform salinity [PSU] (T-only EOS)

    # --- Domain ---
    H_max: float = 4000.0          # Ocean depth [m]

    # --- Vertical grid ---
    n_levels: int = 20             # Number of vertical levels
    dz_surface: float = 10.0       # Top layer thickness [m]
    dz_deep: float = 500.0         # Bottom layer thickness [m]

    # --- Surface restoring ---
    tau_T_days: float = 30.0       # SST restoring timescale [days]
    T_star_eq: float = 25.0        # Equatorial target SST [degC]
    T_star_pole: float = 0.0       # Polar target SST [degC]

    # --- Wind ---
    tau_max: float = 0.1           # Maximum wind stress [Pa]

    # --- Viscosity / diffusion (scaled for ~1 deg) ---
    A_h: float = 2e5               # Laplacian viscosity [m2/s]
    A_v: float = 1e-3              # Vertical viscosity [m2/s]
    K_v: float = 1e-5              # Background vertical diffusivity [m2/s]
    bottom_drag_coeff: float = 1.1e-3  # Linear bottom drag [m/s]

    # --- EOS ---
    alpha_T: float = 2.0e-4        # Thermal expansion [1/K]
    T_ref: float = 10.0            # Reference temperature for EOS [degC]

    # --- Continent geometry (same as global_barotropic_wind) ---
    continent_lon_west: float = 20.0
    continent_lon_east: float = 60.0
    continent_lat_south: float = -55.0
    polar_cap_lat: float = 80.0


def _add_stratification(state, z_coord, config: GlobalOverturningConfig):
    """Add exponential temperature stratification to a uniform state.

    T(z) = T_deep + (T_surface - T_deep) * exp(z / scale_depth)

    where z is negative (depth below surface).
    """
    z_full = np.asarray(z_coord.z_full_ref)   # negative values
    decay = np.exp(z_full / config.T_scale_depth)
    T_profile = config.T_deep + (config.T_surface - config.T_deep) * decay

    T_data = np.array(state.T.data)
    for k in range(z_coord.n_levels):
        T_data[..., k] = T_profile[k]

    return state._replace(
        T=Field(jnp.array(T_data), name="T",
                dims=state.T.dims, units=state.T.units))


def create_initial_conditions(grid_type: str, grid, z_coord,
                              config: GlobalOverturningConfig = None):
    """Create global overturning initial conditions.

    Creates a rest state with exponential T stratification, uniform S,
    and a simplified continent mask (shared with global_barotropic_wind).
    """
    if config is None:
        config = GlobalOverturningConfig()

    from legoesm.ocean.experiments.global_barotropic_wind import (
        create_simplified_continent_mask,
    )

    if grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_surface=config.T_surface, T_deep=config.T_surface,
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
            T_surface=config.T_surface, T_deep=config.T_surface,
            S_uniform=config.S_uniform,
        )
        lon_deg = np.asarray(grid.lonCell, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.latCell, dtype=np.float64) * 180 / np.pi

    else:
        raise NotImplementedError(
            f"Global overturning not implemented for {grid_type}")

    # Apply continent mask
    mask = create_simplified_continent_mask(lon_deg, lat_deg, config)
    mask_typed = np.asarray(mask).astype(np.asarray(state.eta.data).dtype)
    if grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import replace_land_mask
        state = replace_land_mask(state, jnp.array(mask_typed))
    else:
        state = state._replace(land_mask=Field(data=jnp.array(mask_typed)))

    # Add exponential stratification
    state = _add_stratification(state, z_coord, config)

    return state


def create_forcings(grid_type: str, grid,
                    config: GlobalOverturningConfig = None):
    """Create physics config: global wind + surface T restoring.

    Uses ``scheme="combined"`` to apply prescribed 3-belt wind stress
    together with SST restoring toward a cosine-latitude profile.
    """
    if config is None:
        config = GlobalOverturningConfig()

    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.surface_forcing.config import (
        PrescribedForcingConfig, RestoringConfig, SurfaceForcingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import (
        ConstantVerticalMixingConfig, VerticalMixingConfig,
    )
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.convection.config import (
        EnhancedDiffusionConfig, OceanConvectionConfig,
    )

    tau_T_seconds = config.tau_T_days * 86400.0

    return OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="combined",
            prescribed=PrescribedForcingConfig(
                wind_profile="two_belt",
                tau_max=config.tau_max,
            ),
            restoring=RestoringConfig(
                tau_T=tau_T_seconds,
                tau_S=1e30,  # effectively disable salinity restoring
                T_star_eq=config.T_star_eq,
                T_star_pole=config.T_star_pole,
                S_star=config.S_uniform,
                T_profile="cosine",
            ),
        ),
        vertical_mixing=VerticalMixingConfig(
            scheme="constant",
            constant=ConstantVerticalMixingConfig(
                A_v=config.A_v,
                K_v=config.K_v,
            ),
        ),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        convection=OceanConvectionConfig(
            scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(
                K_conv=1.0,       # 1 m2/s where statically unstable
                K_bg=1e-5,        # background (matches K_v)
            ),
        ),
        shortwave_penetration=None,
    )


def create_eos_config(config: GlobalOverturningConfig = None):
    """Return linear EOS config (temperature-only buoyancy, beta_S=0)."""
    if config is None:
        config = GlobalOverturningConfig()
    return LinearEOSConfig(
        rho_ref=1025.0,
        alpha_T=config.alpha_T,
        beta_S=0.0,          # T-only buoyancy (Wolfe & Cessi use b = alpha*g*T)
        T_ref=config.T_ref,
        S_ref=config.S_uniform,
    )


def create_domain_config(config: GlobalOverturningConfig = None) -> Dict[str, Any]:
    if config is None:
        config = GlobalOverturningConfig()
    return {
        "H_max": config.H_max,
        "n_levels": config.n_levels,
        "dz_surface": config.dz_surface,
        "dz_deep": config.dz_deep,
        "description": ("Global baroclinic ocean with stratification, "
                        "SST restoring, and simplified continent"),
        "forcing_type": "global_wind_plus_SST_restoring",
        "stratification": "exponential",
        "eos": "linear (T-only)",
    }


def validate_results(final_state, diagnostics: Dict[str, list],
                     config: GlobalOverturningConfig = None) -> Tuple[bool, str]:
    if config is None:
        config = GlobalOverturningConfig()

    success = True
    notes_parts = []

    # Check finiteness
    for field_name in ("eta", "T", "u"):
        if hasattr(final_state, field_name):
            data = getattr(final_state, field_name).data
            if not jnp.all(jnp.isfinite(data)):
                return False, f"NaN/Inf in final {field_name}"

    # Max speed
    max_speed_list = diagnostics.get("max_speed", [])
    if max_speed_list:
        speed = max_speed_list[-1]
        notes_parts.append(f"max_speed={speed:.4f}m/s")
        if speed < 0.01:
            notes_parts.append("WARN: weak circulation")
        elif speed > 5.0:
            success = False
            notes_parts.append("FAIL: excessive speed")

    # SSH drift
    eta_list = diagnostics.get("mean_eta", [])
    if len(eta_list) >= 2:
        eta_drift = abs(eta_list[-1] - eta_list[0])
        notes_parts.append(f"eta_drift={eta_drift:.2e}")

    # Stratification check: surface T should be warmer than deep T
    mean_T_list = diagnostics.get("mean_T", [])
    if mean_T_list:
        notes_parts.append(f"mean_T={mean_T_list[-1]:.2f}degC")

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
    "name": "global_overturning",
    "description": ("Global baroclinic overturning circulation with "
                    "stratification and SST restoring (Wolfe & Cessi 2010 inspired)"),
    "scientific_purpose": ("Validates global MOC, thermocline, gyres, "
                           "and ACC as stepping stone toward OMIP"),
    "reference": "Wolfe & Cessi (2010, JPO)",
    "config_class": GlobalOverturningConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": create_domain_config,
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 90.0,   # days — longer for baroclinic adjustment
    "quick_duration": 5.0,      # days
    "grid_support": {
        "cubed_sphere": False,  # face-boundary instability (#100)
        "latlon": True,
        "mpas": True,
        "spectral": False,
    },
}
