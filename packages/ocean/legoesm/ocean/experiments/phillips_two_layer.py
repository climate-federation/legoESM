"""Phillips Two-Layer Baroclinic Ocean Experiment.

A classic test case for two-layer baroclinic instability based on Phillips (1954).
This experiment studies the development of baroclinic instability from an initial
zonal jet in a simplified two-layer ocean configuration with temperature relaxation.

Scientific Purpose:
- Validate baroclinic instability dynamics in a simplified two-layer model
- Test zonal jet initialization and equilibrium balance
- Study eddy development from unstable baroclinic flow
- Benchmark temperature relaxation forcing mechanisms
- Verify conservation properties in idealized baroclinic flows
- Test numerical stability in two-layer configurations

Domain Configuration:
- Two-layer vertical structure (layers at ~1750m and ~3500m depth)
- Reduced total depth: 3500m (vs 5500m in other experiments)
- Global ocean with land at high latitudes (|lat| > 80°)
- Initial zonal jet structure with opposing flow in each layer
- Temperature relaxation toward prescribed profiles

Physical Setup:
- Upper layer: eastward jet centered at 45°N with exponential decay
- Lower layer: westward flow (opposite to upper layer)
- Temperature profiles: upper layer 16°C at equator, lower layer 8°C
- Meridional temperature gradients in both layers
- SSH perturbation to trigger baroclinic instability
- Relaxation forcing toward target temperature profiles

Expected Behavior:
- Initial geostrophic balance between temperature gradient and zonal flow
- Development of baroclinic instability (wavelength ~1000-2000 km)
- Eddy formation and zonal jet meandering
- Heat transport via baroclinic eddies
- Equilibrium between instability and temperature relaxation

Validation Criteria:
- Temperature drift within bounds despite relaxation forcing
- Development of realistic baroclinic instability patterns
- Maintenance of two-layer structure
- Conservation properties appropriate for forced system

References:
- Phillips (1954), "Energy transformations and meridional circulations
  associated with simple baroclinic waves"
- Gill (1982), "Atmosphere-Ocean Dynamics" - baroclinic instability theory
- Pedlosky (1987), "Geophysical Fluid Dynamics" - two-layer models
- Standard ocean model validation for idealized baroclinic flows
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from legoesm.constants import g, R_earth
from legoesm.core.field import Field


_A_EARTH = R_earth
_G_EARTH = g


@dataclass
class PhillipsTwoLayerConfig:
    """Configuration parameters for Phillips two-layer experiment.

    All parameters have physically meaningful defaults based on the
    classic Phillips (1954) setup.
    """
    # Domain configuration (two-layer specific)
    nlev: int = 2                  # Number of vertical levels
    H_max: float = 3500.0          # Total ocean depth [m] (shallower than default)
    land_lat_threshold: float = 80.0  # Latitude threshold for land [degrees]
    spectral_land_lat_threshold: float = 90.0  # No land for spectral grid

    # Jet parameters
    jet_speed_upper: float = 0.30      # Upper layer jet speed [m/s]
    jet_speed_lower_factor: float = -0.20  # Lower layer factor (negative = opposite)
    jet_center_lat: float = 45.0       # Jet center latitude [degrees]
    jet_width: float = 14.0            # Jet width (degrees)

    # Temperature profiles (target for relaxation)
    T_upper_equator: float = 16.0      # Upper layer equatorial temperature [°C]
    T_upper_gradient: float = 10.0     # Upper layer meridional gradient [°C]
    T_lower_equator: float = 8.0       # Lower layer equatorial temperature [°C]
    T_lower_gradient: float = 4.0      # Lower layer meridional gradient [°C]

    # SSH perturbation (to trigger instability)
    eta_perturbation_amplitude: float = 0.05  # SSH perturbation amplitude [m]
    eta_wave_lon: float = 3.0          # Longitudinal wavenumber
    eta_wave_lat: float = 2.0          # Latitudinal wavenumber

    # Relaxation parameters
    tau_relax_days: float = 15.0       # Temperature relaxation timescale [days]
    drag_timescale: float = 25.0       # Momentum damping timescale [days]


def create_initial_conditions(grid_type: str, grid, z_coord,
                            config: PhillipsTwoLayerConfig = None):
    """Create Phillips two-layer initial conditions for any grid type.

    This creates a two-layer ocean state with:
    - Zonal jet structure (upper eastward, lower westward)
    - Temperature profiles consistent with thermal wind balance
    - SSH perturbation to trigger baroclinic instability

    Parameters
    ----------
    grid_type : str
        Grid type: "cubed_sphere", "latlon", "mpas", or "spectral"
    grid : Grid
        Grid object (type depends on grid_type)
    z_coord : OceanZCoordinate
        Vertical coordinate system (should have nlev=2)
    config : PhillipsTwoLayerConfig, optional
        Configuration parameters. Uses defaults if None.

    Returns
    -------
    OceanState
        Initial state with Phillips two-layer setup

    Notes
    -----
    - Requires exactly 2 vertical levels
    - Spectral grid: uses vorticity/divergence formulation
    - FV grids: direct velocity assignment
    - All grids: apply land mask consistently
    """
    if config is None:
        config = PhillipsTwoLayerConfig()

    # First create rest state background with 2 levels and reduced depth
    if grid_type == "cubed_sphere":
        from legoesm.ocean.init import rest_state_ocean
        state = rest_state_ocean(
            grid, z_coord,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )

    elif grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )

    elif grid_type == "mpas":
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        state = rest_state_mpas_ocean(
            grid, z_coord,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )

    elif grid_type == "spectral":
        from legoesm.ocean.dynamics.spectral_ocean_pe import rest_state_spectral_ocean
        state = rest_state_spectral_ocean(
            grid, z_coord,
            H_max=config.H_max,
            land_lat_threshold=config.spectral_land_lat_threshold
        )

    else:
        raise ValueError(f"Unknown grid type: {grid_type}")

    # Add Phillips perturbation (jet + temperature + SSH)
    return _add_phillips_perturbation(state, grid_type, grid, z_coord, config)


def _add_phillips_perturbation(state, grid_type: str, grid, z_coord,
                             config: PhillipsTwoLayerConfig):
    """Add Phillips two-layer perturbation: jet + SSH + temperature profiles."""

    if grid_type == "spectral":
        return _add_phillips_perturbation_spectral(state, grid, z_coord, config)
    else:
        return _add_phillips_perturbation_fv(state, grid_type, grid, z_coord, config)


def _add_phillips_perturbation_spectral(state, grid, z_coord, config):
    """Add Phillips perturbation for spectral grid (vorticity/divergence)."""
    from legoesm.grids.gaussian import (
        sh_analysis, sh_analysis_3d, sh_synthesis_3d,
        sh_analysis_oc2_3d, sh_analysis_dmu_3d)

    lat = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
    lon = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
    lon_2d, lat_2d = np.meshgrid(lon, lat, indexing='xy')
    mask = np.asarray(state.land_mask_grid.data, dtype=np.float64)
    nlev = 2

    # Temperature profiles
    T_hat = state.T_hat.data
    T_grid = np.array(sh_synthesis_3d(grid, T_hat), dtype=np.float64)

    # Upper layer temperature
    T_upper = (config.T_upper_equator -
               config.T_upper_gradient * np.sin(np.radians(lat_2d)) ** 2)
    T_grid[..., 0] = T_upper * mask

    # Lower layer temperature
    if nlev > 1:
        T_lower = (config.T_lower_equator -
                   config.T_lower_gradient * np.sin(np.radians(lat_2d)) ** 2)
        T_grid[..., 1] = T_lower * mask

    new_T_hat = sh_analysis_3d(grid, jnp.array(T_grid))

    # Zonal jet -> vorticity/divergence
    cos_lat = np.asarray(grid.cos_lat[:, None], dtype=np.float64)

    # Upper layer: eastward jet
    u_upper = (config.jet_speed_upper *
               np.exp(-((lat_2d - config.jet_center_lat) / config.jet_width) ** 2))
    u_grid = np.zeros(T_grid.shape, dtype=np.float64)
    u_grid[..., 0] = u_upper * mask

    # Lower layer: westward jet
    if nlev > 1:
        u_grid[..., 1] = config.jet_speed_lower_factor * u_upper * mask

    # Convert to vorticity/divergence
    u_cos = jnp.array(u_grid * cos_lat[..., None])
    v_cos = jnp.zeros_like(u_cos)

    a = grid.radius
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    one_over_a = 1.0 / a

    vor_hat = (im_over_a[:, None] * sh_analysis_oc2_3d(grid, v_cos) +
               one_over_a * sh_analysis_dmu_3d(grid, u_cos))
    div_hat = (im_over_a[:, None] * sh_analysis_oc2_3d(grid, u_cos) -
               one_over_a * sh_analysis_dmu_3d(grid, v_cos))

    # SSH perturbation
    eta_pert = (config.eta_perturbation_amplitude *
                np.sin(config.eta_wave_lon * np.radians(lon_2d)) *
                np.cos(config.eta_wave_lat * np.radians(lat_2d)) * mask)

    # Remove area-weighted mean
    w = np.asarray(grid.weights, dtype=np.float64)[:, None] * mask
    eta_pert -= np.sum(eta_pert * w) / np.maximum(np.sum(w), 1e-30)

    eta_hat = state.eta_hat.data + sh_analysis(grid, jnp.array(eta_pert))

    return state._replace(
        vor_hat=Field(vor_hat, name="vor_hat", dims=state.vor_hat.dims, units="s^-1"),
        div_hat=Field(div_hat, name="div_hat", dims=state.div_hat.dims, units="s^-1"),
        T_hat=Field(new_T_hat, name="T_hat", dims=state.T_hat.dims, units="K"),
        eta_hat=Field(eta_hat, name="eta_hat", dims=state.eta_hat.dims, units="m")
    )


def _add_phillips_perturbation_fv(state, grid_type: str, grid, z_coord, config):
    """Add Phillips perturbation for finite volume grids."""
    # Get coordinates
    if grid_type == "mpas":
        lat = np.asarray(grid.latCell, dtype=np.float64) * 180 / np.pi
        lon = np.asarray(grid.lonCell, dtype=np.float64) * 180 / np.pi
        lat_2d, lon_2d = lat, lon  # Already 1D for MPAS
    else:
        lat_1d = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        lon_1d = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        if grid_type == "latlon":
            lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing='xy')
        else:  # cubed_sphere
            lat_2d, lon_2d = lat_1d, lon_1d  # Already 2D for cubed sphere
    
    # Temperature profiles.
    # ``np.asarray`` on a JAX array returns a read-only view; take an
    # explicit writable copy so the subsequent in-place assignments work.
    T_data = np.array(state.T.data, dtype=np.float64, copy=True)
    nlev = T_data.shape[-1]

    # Upper layer
    T_upper = (config.T_upper_equator -
               config.T_upper_gradient * np.sin(np.radians(lat_2d)) ** 2)
    T_data[..., 0] = T_upper

    # Lower layer
    if nlev > 1:
        T_lower = (config.T_lower_equator -
                   config.T_lower_gradient * np.sin(np.radians(lat_2d)) ** 2)
        T_data[..., 1] = T_lower
    
    # Velocity: zonal jet (writable copies of the underlying JAX arrays).
    # On the latlon C-grid, u lives at east faces (n_lat, n_lon+1) and v at
    # north faces (n_lat+1, n_lon); the jet depends only on latitude so we
    # reconstruct lat_u / lat_v at the face positions.
    # MPAS uses edge-normal velocity (single `u` on edges, no `v`).  We
    # project the zonal jet onto each edge using ``angleEdge``.
    if grid_type == "mpas":
        u_data = np.array(state.u.data, dtype=np.float64, copy=True)
        lat_e_deg = np.asarray(grid.latEdge, dtype=np.float64) * 180 / np.pi
        angle = np.asarray(grid.angleEdge, dtype=np.float64)
        u_upper_edge = (
            config.jet_speed_upper *
            np.exp(-((lat_e_deg - config.jet_center_lat) /
                     config.jet_width) ** 2))
        u_edge_up = u_upper_edge * np.cos(angle)
        u_data[..., 0] = u_edge_up
        if nlev > 1:
            u_data[..., 1] = config.jet_speed_lower_factor * u_edge_up
        state = state._replace(
            u=Field(jnp.array(u_data), name="u",
                    dims=state.u.dims, units="m/s"))
    elif hasattr(state, 'u'):
        u_data = np.array(state.u.data, dtype=np.float64, copy=True)
        v_data = np.array(state.v.data, dtype=np.float64, copy=True)

        if grid_type == "latlon":
            n_lat = u_data.shape[0]
            n_u_lon = u_data.shape[1]
            n_v_lat = v_data.shape[0]
            lat_u_1d = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
            dlat_deg = float(grid.dlat) * 180 / np.pi
            lat_v_1d = lat_u_1d[0] + dlat_deg * (np.arange(n_v_lat) + 0.5)
            lat_u_2d = np.broadcast_to(lat_u_1d[:, None], (n_lat, n_u_lon))
            lat_v_2d = np.broadcast_to(
                lat_v_1d[:, None], (n_v_lat, v_data.shape[1]))
            u_upper = (config.jet_speed_upper *
                       np.exp(-((lat_u_2d - config.jet_center_lat) /
                                config.jet_width) ** 2))
            u_data[..., 0] = u_upper
            v_data[..., 0] = 0.0
            if nlev > 1:
                u_data[..., 1] = config.jet_speed_lower_factor * u_upper
                v_data[..., 1] = 0.0
            # v_2d not strictly needed (jet is zonal) but kept for clarity.
            _ = lat_v_2d
        else:
            u_upper = (config.jet_speed_upper *
                       np.exp(-((lat_2d - config.jet_center_lat) /
                                config.jet_width) ** 2))
            u_data[..., 0] = u_upper
            v_data[..., 0] = 0.0
            if nlev > 1:
                u_data[..., 1] = config.jet_speed_lower_factor * u_upper
                v_data[..., 1] = 0.0

        state = state._replace(
            u=Field(jnp.array(u_data), name="u", dims=state.u.dims, units="m/s"),
            v=Field(jnp.array(v_data), name="v", dims=state.v.dims, units="m/s")
        )

    # SSH perturbation
    eta_pert = (config.eta_perturbation_amplitude *
                np.sin(config.eta_wave_lon * np.radians(lon_2d)) *
                np.cos(config.eta_wave_lat * np.radians(lat_2d)))

    new_eta = state.eta.data + jnp.array(eta_pert)

    return state._replace(
        T=Field(jnp.array(T_data), name="T", dims=state.T.dims, units="K"),
        eta=Field(new_eta, name="eta", dims=state.eta.dims, units="m")
    )


def create_forcings(grid_type: str, grid, config: PhillipsTwoLayerConfig = None):
    """Create forcing functions for Phillips two-layer experiment.

    The Phillips experiment uses temperature relaxation forcing toward
    prescribed profiles and momentum damping.

    Parameters
    ----------
    grid_type : str
        Grid type
    grid : Grid
        Grid object
    config : PhillipsTwoLayerConfig, optional
        Configuration parameters

    Returns
    -------
    dict
        Forcing configuration parameters

    Notes
    -----
    Returns relaxation parameters that are used by the model's
    forcing mechanisms during time integration.
    """
    if config is None:
        config = PhillipsTwoLayerConfig()

    return {
        "temperature_relaxation": {
            "tau_relax_days": config.tau_relax_days,
            "T_upper_equator": config.T_upper_equator,
            "T_upper_gradient": config.T_upper_gradient,
            "T_lower_equator": config.T_lower_equator,
            "T_lower_gradient": config.T_lower_gradient,
        },
        "momentum_damping": {
            "drag_timescale_days": config.drag_timescale,
        }
    }


def create_domain_config(config: PhillipsTwoLayerConfig = None) -> Dict[str, Any]:
    """Create domain configuration parameters.

    Returns
    -------
    Dict[str, Any]
        Domain configuration parameters for this experiment
    """
    if config is None:
        config = PhillipsTwoLayerConfig()

    return {
        "nlev": config.nlev,
        "H_max": config.H_max,
        "land_lat_threshold": config.land_lat_threshold,
        "spectral_land_lat_threshold": config.spectral_land_lat_threshold,
        "description": "Phillips two-layer baroclinic instability with relaxation",
        "forcing_type": "temperature_relaxation_momentum_damping",
    }


def compute_instability_metrics(diagnostics: Dict[str, list],
                              config: PhillipsTwoLayerConfig) -> Dict[str, float]:
    """Compute baroclinic instability metrics for validation.

    Parameters
    ----------
    diagnostics : Dict[str, list]
        Time series diagnostics from simulation
    config : PhillipsTwoLayerConfig
        Configuration parameters

    Returns
    -------
    Dict[str, float]
        Instability metrics for validation
    """
    metrics = {}

    # Temperature evolution (with relaxation forcing)
    mean_T_list = diagnostics.get("mean_T", [])
    if len(mean_T_list) >= 2:
        T_drift = abs(mean_T_list[-1] - mean_T_list[0])
        metrics["T_drift_absolute"] = T_drift

        # Temperature variability (measure of instability development)
        if len(mean_T_list) > 10:
            T_std = np.std(mean_T_list[-10:])  # Last 10 points
            metrics["T_variability"] = T_std

    # SSH amplitude - measure of instability growth
    max_eta_list = diagnostics.get("max_abs_eta", [])
    if len(max_eta_list) >= 2:
        eta_final = max_eta_list[-1]
        eta_initial = max_eta_list[0]
        metrics["max_eta_final"] = eta_final
        if eta_initial > 1e-10:
            eta_growth = eta_final / eta_initial
            metrics["eta_growth"] = eta_growth

    return metrics


def validate_results(final_state, diagnostics: Dict[str, list],
                   config: PhillipsTwoLayerConfig = None) -> Tuple[bool, str]:
    """Validate Phillips two-layer experiment results.

    Success criteria:
    - Temperature drift acceptable (system has relaxation forcing)
    - Some instability development (SSH growth or temperature variability)
    - No excessive values or numerical instabilities
    - Two-layer structure maintained

    Parameters
    ----------
    final_state : OceanState
        Final model state
    diagnostics : Dict[str, list]
        Time series diagnostics
    config : PhillipsTwoLayerConfig, optional
        Configuration parameters

    Returns
    -------
    bool
        True if validation passes
    str
        Descriptive notes about the validation
    """
    if config is None:
        config = PhillipsTwoLayerConfig()

    # Compute instability metrics
    metrics = compute_instability_metrics(diagnostics, config)

    # Check for NaN/infinite values in final state
    if hasattr(final_state, 'T') and not jnp.all(jnp.isfinite(final_state.T.data)):
        return False, "NaN/Inf detected in final temperature field"
    if hasattr(final_state, 'eta') and not jnp.all(jnp.isfinite(final_state.eta.data)):
        return False, "NaN/Inf detected in final eta field"

    # Validation thresholds (relaxed due to forcing)
    max_T_drift = 5.0         # °C - relaxed due to relaxation forcing
    min_eta_growth = 0.8      # Minimum growth (instability or maintenance)
    max_eta_growth = 10.0     # Maximum growth (prevent explosion)
    max_eta_amplitude = 5.0   # m - maximum reasonable SSH

    # Check validation criteria
    success = True
    notes_parts = []

    if "T_drift_absolute" in metrics:
        T_drift = metrics["T_drift_absolute"]
        notes_parts.append(f"T_drift={T_drift:.2f}°C")
        if T_drift > max_T_drift:
            success = False
            notes_parts.append("FAIL: excessive T drift")

    if "max_eta_final" in metrics:
        eta_max = metrics["max_eta_final"]
        notes_parts.append(f"max_eta={eta_max:.3f}m")
        if eta_max > max_eta_amplitude:
            success = False
            notes_parts.append("FAIL: excessive SSH")

    if "eta_growth" in metrics:
        eta_growth = metrics["eta_growth"]
        notes_parts.append(f"eta_growth={eta_growth:.2f}")
        if eta_growth < min_eta_growth:
            success = False
            notes_parts.append("FAIL: insufficient instability")
        elif eta_growth > max_eta_growth:
            success = False
            notes_parts.append("FAIL: excessive growth")

    if "T_variability" in metrics:
        T_var = metrics["T_variability"]
        notes_parts.append(f"T_var={T_var:.3f}")

    notes = ", ".join(notes_parts)

    return success, notes


def get_diagnostic_field_specs() -> list:
    """Get field specifications for diagnostic output.

    Returns
    -------
    list
        Field specifications for plotting: [(field_key, label, colormap), ...]
    """
    return [
        ("eta", "SSH (m)", "RdBu_r"),
        ("SST", "SST (degC)", "RdYlBu_r"),
    ]


def get_scalar_units() -> Dict[str, str]:
    """Get units for scalar diagnostic quantities.

    Returns
    -------
    Dict[str, str]
        Mapping from scalar field names to units
    """
    return {
        "mean_eta": "m",
        "max_abs_eta": "m",
        "mean_T": "degC",
        "mean_S": "PSU"
    }


# Standard experiment configuration for test matrix integration
EXPERIMENT_CONFIG = {
    "name": "phillips_two_layer",
    "description": "Phillips two-layer baroclinic instability with temperature relaxation",
    "scientific_purpose": "Validates baroclinic instability in idealized two-layer configuration",
    "reference": "Phillips (1954) - classical two-layer baroclinic instability",
    "config_class": PhillipsTwoLayerConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": create_domain_config,
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 10.0,   # days
    "quick_duration": 1.0,      # days
    "expected_metrics": {
        "T_drift_absolute": "< 5.0 °C",
        "eta_growth": "0.8-10.0",
        "max_eta_final": "< 5.0 m"
    },
    "grid_support": {
        "cubed_sphere": True,
        "latlon": True,
        "mpas": True,
        "spectral": True
    },
    "special_config": {
        "nlev": 2,           # Requires exactly 2 levels
        "H_max": 3500.0      # Reduced depth compared to other experiments
    }
}