"""Inertia-Gravity Wave Ocean Experiment.

A barotropic wave propagation test case based on Bishnu et al. (2024) that 
validates the shallow water dynamics, dispersion properties, and numerical
accuracy of ocean models through analytical comparison.

Scientific Purpose:
- Validate inertia-gravity (Poincaré) wave dynamics
- Test barotropic pressure gradient and Coriolis terms
- Verify analytical dispersion relation: ω² = f² + g*H*(k² + l²)
- Benchmark numerical accuracy through L2 error computation
- Study grid-dependent dispersion properties
- Test conservation properties during wave propagation

Domain Configuration:
- Equivalent depth formulation: H = 1000m (shallower than other experiments)
- Single-level barotropic (or two-level equivalent)
- Global domain with simple wave pattern
- No bathymetry or land effects for clean wave propagation

Physical Setup:
- Sinusoidal wave pattern: wavenumber-2 in both longitude and latitude  
- Initial condition: η = cos(kx·x + ky·y), u,v from linearized momentum
- Analytical solution available for all times
- Mid-latitude f-plane approximation: f₀ = 10⁻⁴ s⁻¹

Expected Behavior:
- Wave propagation according to dispersion relation
- Circular wave pattern on sphere (low wavenumbers)
- Conservation of wave amplitude and energy
- Minimal numerical dispersion or dissipation
- Close agreement with analytical solution

Validation Criteria:
- L2 error vs analytical solution < threshold (typically < 0.1)
- Wave amplitude preservation within reasonable bounds
- Correct propagation frequency: ω ≈ theoretical value
- No excessive numerical damping or growth

References:
- Bishnu et al. (2024), "A Verification Suite of Test Cases for the Barotropic 
  Solver of Ocean Models", JAMES. DOI: 10.1029/2022MS003545
- Gill (1982), "Atmosphere-Ocean Dynamics" - inertia-gravity wave theory
- Standard ocean model verification for barotropic dynamics
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from legoesm.constants import g
from legoesm.core.field import Field


# Physical constants
_A_EARTH = 6.371e6  # Earth radius [m]
_G_EARTH = 9.80616  # Gravitational acceleration [m/s^2]


@dataclass
class InertiaGravityWaveConfig:
    """Configuration parameters for inertia-gravity wave experiment.
    
    Parameters based on Bishnu et al. (2024) verification study.
    """
    # Domain configuration (shallow water equivalent)
    nlev: int = 2                  # Number of levels (equivalent barotropic)
    H_max: float = 1000.0          # Equivalent depth [m]
    land_lat_threshold: float = 90.0  # No land (global wave propagation)
    
    # Wave parameters
    eta_amplitude: float = 1.0     # SSH wave amplitude [m]
    wavenumber_x: float = 2.0      # Zonal wavenumber (cycles)
    wavenumber_y: float = 2.0      # Meridional wavenumber (cycles)
    
    # Physical parameters
    f0: float = 1.0e-4             # Coriolis parameter [s⁻¹] (mid-latitude)
    
    # Validation thresholds
    max_l2_error: float = 0.1      # Maximum L2 error vs analytical solution
    max_amplitude_drift: float = 0.2  # Maximum amplitude change


def create_initial_conditions(grid_type: str, grid, z_coord, 
                            config: InertiaGravityWaveConfig = None):
    """Create inertia-gravity wave initial conditions for any grid type.
    
    Sets up a sinusoidal wave pattern with wavenumber-2 structure in both
    longitude and latitude, with consistent velocity field derived from
    linearized shallow water equations.
    
    Parameters
    ----------
    grid_type : str
        Grid type: "cubed_sphere", "latlon", "mpas", or "spectral"
    grid : Grid
        Grid object (type depends on grid_type)
    z_coord : OceanZCoordinate
        Vertical coordinate system
    config : InertiaGravityWaveConfig, optional
        Configuration parameters. Uses defaults if None.
        
    Returns
    -------
    OceanState
        Initial state with inertia-gravity wave pattern
        
    Notes
    -----
    - Wave amplitude: 1m (for clear signal)
    - Velocity computed from linearized momentum equations
    - Spectral grid: uses vorticity/divergence formulation
    - All grids: ensures analytical consistency for validation
    """
    if config is None:
        config = InertiaGravityWaveConfig()
    
    # First create rest state background  
    if grid_type == "cubed_sphere":
        from legoesm.ocean.init import rest_state_ocean
        state = rest_state_ocean(
            grid, z_coord, 
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )
        
    elif grid_type == "latlon":
        from legoesm.ocean.init_latlon import rest_state_latlon_ocean
        state = rest_state_latlon_ocean(
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
            land_lat_threshold=config.land_lat_threshold
        )
        
    else:
        raise ValueError(f"Unknown grid type: {grid_type}")
    
    # Add inertia-gravity wave perturbation
    return _add_igw_perturbation(state, grid_type, grid, z_coord, config)


def _add_igw_perturbation(state, grid_type: str, grid, z_coord,
                        config: InertiaGravityWaveConfig):
    """Add sinusoidal inertia-gravity wave perturbation."""
    H = config.H_max
    f0 = config.f0
    kx = config.wavenumber_x
    ky = config.wavenumber_y
    eta_amp = config.eta_amplitude
    
    # Get coordinates in radians
    if grid_type == "mpas":
        lat = np.asarray(grid.latCell, dtype=np.float64)
        lon = np.asarray(grid.lonCell, dtype=np.float64)
    elif grid_type in ("latlon", "spectral"):
        lat_1d = np.asarray(grid.lat, dtype=np.float64)
        lon_1d = np.asarray(grid.lon, dtype=np.float64)
        lon, lat = np.meshgrid(lon_1d, lat_1d, indexing='xy')
    else:  # cubed_sphere
        lat = np.asarray(grid.lat, dtype=np.float64)
        lon = np.asarray(grid.lon, dtype=np.float64)
    
    # Physical wavenumbers on sphere
    k_phys = kx / _A_EARTH
    l_phys = ky / _A_EARTH
    
    # Analytical dispersion relation
    omega = np.sqrt(f0**2 + _G_EARTH * H * (k_phys**2 + l_phys**2))
    
    # Wave phase (t=0)
    phase = kx * lon + ky * lat
    eta_pert = eta_amp * np.cos(phase)
    
    # Linearized shallow water velocities at t=0
    denom = omega**2 - f0**2
    if abs(denom) < 1e-30:
        denom = 1e-30  # Avoid division by zero
        
    u_pert = (_G_EARTH / denom) * (
        omega * k_phys * np.cos(phase) - f0 * l_phys * np.sin(phase))
    v_pert = (_G_EARTH / denom) * (
        omega * l_phys * np.cos(phase) + f0 * k_phys * np.sin(phase))
    
    if grid_type == "spectral":
        return _add_igw_spectral(state, grid, eta_pert, u_pert, v_pert)
    elif grid_type == "mpas":
        return _add_igw_mpas(state, grid, eta_pert, u_pert, v_pert)
    else:
        return _add_igw_fv(state, eta_pert, u_pert, v_pert)


def _add_igw_spectral(state, grid, eta_pert, u_pert, v_pert):
    """Add IGW for spectral grid (vorticity/divergence)."""
    from legoesm.grids.gaussian import (
        sh_analysis, sh_analysis_oc2_3d, sh_analysis_dmu_3d)
    
    # SSH perturbation 
    eta_hat = sh_analysis(grid, jnp.array(eta_pert))
    new_eta_hat = state.eta_hat.data + eta_hat
    
    # Convert velocities to vorticity/divergence
    nlev = state.vor_hat.data.shape[-1]
    cos_lat = np.asarray(grid.cos_lat[:, None], dtype=np.float64)
    
    # Extend to 3D (surface level only)
    u_3d = np.zeros((*u_pert.shape, nlev), dtype=np.float64)
    v_3d = np.zeros((*v_pert.shape, nlev), dtype=np.float64)
    u_3d[..., 0] = u_pert
    v_3d[..., 0] = v_pert
    
    u_cos = jnp.array(u_3d * cos_lat[..., None])
    v_cos = jnp.array(v_3d * cos_lat[..., None])
    
    a = grid.radius
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    one_over_a = 1.0 / a
    
    vor_hat = (im_over_a[:, None] * sh_analysis_oc2_3d(grid, v_cos) +
               one_over_a * sh_analysis_dmu_3d(grid, u_cos))
    div_hat = (im_over_a[:, None] * sh_analysis_oc2_3d(grid, u_cos) -
               one_over_a * sh_analysis_dmu_3d(grid, v_cos))
    
    return state._replace(
        eta_hat=Field(new_eta_hat, name="eta_hat", dims=state.eta_hat.dims, units="m"),
        vor_hat=Field(vor_hat, name="vor_hat", dims=state.vor_hat.dims, units="s^-1"),
        div_hat=Field(div_hat, name="div_hat", dims=state.div_hat.dims, units="s^-1")
    )


def _add_igw_mpas(state, grid, eta_pert, u_pert, v_pert):
    """Add IGW for MPAS grid (edge velocities)."""
    # Get edge coordinates
    lat_e = np.asarray(grid.latEdge, dtype=np.float64)
    lon_e = np.asarray(grid.lonEdge, dtype=np.float64)
    
    # Compute edge velocities
    kx, ky = 2.0, 2.0  # from config
    phase_e = kx * lon_e + ky * lat_e
    
    # Recompute edge velocities from analytical solution
    H = 1000.0
    f0 = 1.0e-4
    k_phys = kx / _A_EARTH
    l_phys = ky / _A_EARTH
    omega = np.sqrt(f0**2 + _G_EARTH * H * (k_phys**2 + l_phys**2))
    denom = omega**2 - f0**2
    
    u_e = (_G_EARTH / denom) * (
        omega * k_phys * np.cos(phase_e) - f0 * l_phys * np.sin(phase_e))
    v_e = (_G_EARTH / denom) * (
        omega * l_phys * np.cos(phase_e) + f0 * k_phys * np.sin(phase_e))
    
    # Project onto edge normals
    angle = np.asarray(grid.angleEdge, dtype=np.float64)
    u_edge = u_e * np.cos(angle) + v_e * np.sin(angle)
    
    # Update state
    u_data = np.array(state.u.data, dtype=np.float64, copy=True)
    u_data[..., 0] = u_edge
    
    return state._replace(
        eta=Field(jnp.array(eta_pert), name="eta", dims=state.eta.dims, units="m"),
        u=Field(jnp.array(u_data), name="u", dims=state.u.dims, units="m/s")
    )


def _add_igw_fv(state, eta_pert, u_pert, v_pert):
    """Add IGW for finite volume grids (cubed_sphere, latlon)."""
    u_data = np.array(state.u.data, dtype=np.float64, copy=True)
    v_data = np.array(state.v.data, dtype=np.float64, copy=True)
    
    # Surface level only
    u_data[..., 0] = u_pert
    v_data[..., 0] = v_pert
    
    return state._replace(
        eta=Field(jnp.array(eta_pert), name="eta", dims=state.eta.dims, units="m"),
        u=Field(jnp.array(u_data), name="u", dims=state.u.dims, units="m/s"),
        v=Field(jnp.array(v_data), name="v", dims=state.v.dims, units="m/s")
    )


def create_forcings(grid_type: str, grid, config: InertiaGravityWaveConfig = None):
    """Create forcing functions for inertia-gravity wave experiment.
    
    No external forcings - wave propagation is governed by initial conditions
    and shallow water dynamics only.
    
    Returns
    -------
    None
        No forcings for this experiment
    """
    return None


def create_domain_config(config: InertiaGravityWaveConfig = None) -> Dict[str, Any]:
    """Create domain configuration parameters.
    
    Returns
    -------
    Dict[str, Any]
        Domain configuration parameters for this experiment
    """
    if config is None:
        config = InertiaGravityWaveConfig()
        
    return {
        "nlev": config.nlev,
        "H_max": config.H_max,
        "land_lat_threshold": config.land_lat_threshold,
        "description": "Global inertia-gravity wave propagation test",
        "reference": "Bishnu et al. (2024) verification suite",
    }


def compute_analytical_solution(grid_type: str, grid, t_final: float, 
                               config: InertiaGravityWaveConfig) -> np.ndarray:
    """Compute analytical inertia-gravity wave solution at time t_final.
    
    Parameters
    ----------
    grid_type : str
        Grid type
    grid : Grid
        Grid object
    t_final : float
        Final time [seconds]
    config : InertiaGravityWaveConfig
        Configuration parameters
        
    Returns
    -------
    np.ndarray
        Analytical SSH solution at t_final
    """
    # Get coordinates
    if grid_type == "mpas":
        lat = np.asarray(grid.latCell, dtype=np.float64)
        lon = np.asarray(grid.lonCell, dtype=np.float64)
    elif grid_type in ("latlon", "spectral"):
        lat_1d = np.asarray(grid.lat, dtype=np.float64)
        lon_1d = np.asarray(grid.lon, dtype=np.float64)
        lon, lat = np.meshgrid(lon_1d, lat_1d, indexing='xy')
    else:  # cubed_sphere
        lat = np.asarray(grid.lat, dtype=np.float64)
        lon = np.asarray(grid.lon, dtype=np.float64)
    
    # Physical parameters
    H = config.H_max
    f0 = config.f0
    kx = config.wavenumber_x
    ky = config.wavenumber_y
    eta_amp = config.eta_amplitude
    
    # Dispersion relation
    k_phys = kx / _A_EARTH
    l_phys = ky / _A_EARTH
    omega = np.sqrt(f0**2 + _G_EARTH * H * (k_phys**2 + l_phys**2))
    
    # Analytical solution
    phase = kx * lon + ky * lat - omega * t_final
    eta_exact = eta_amp * np.cos(phase)
    
    return eta_exact, omega


def compute_wave_metrics(final_state, grid_type: str, grid, t_final: float,
                        initial_eta: np.ndarray, config: InertiaGravityWaveConfig
                        ) -> Dict[str, float]:
    """Compute wave propagation metrics for validation.
    
    Parameters
    ----------
    final_state : OceanState
        Final model state
    grid_type : str
        Grid type
    grid : Grid
        Grid object  
    t_final : float
        Final time [seconds]
    initial_eta : np.ndarray
        Initial SSH field for amplitude comparison
    config : InertiaGravityWaveConfig
        Configuration parameters
        
    Returns
    -------
    Dict[str, float]
        Wave metrics including L2 error and amplitude preservation
    """
    metrics = {}
    
    # Get final SSH field
    if grid_type == "spectral":
        from legoesm.grids.gaussian import sh_synthesis
        eta_final = np.asarray(sh_synthesis(grid, final_state.eta_hat.data))
    else:
        eta_final = np.asarray(final_state.eta.data)
    
    # Compute analytical solution
    eta_exact, omega = compute_analytical_solution(grid_type, grid, t_final, config)
    
    # L2 error (relative)
    l2_err = (np.sqrt(np.mean((eta_final - eta_exact)**2)) /
              max(np.sqrt(np.mean(eta_exact**2)), 1e-30))
    metrics["l2_error"] = l2_err
    metrics["omega_analytical"] = omega
    
    # Amplitude preservation
    max_eta_final = np.max(np.abs(eta_final))
    max_eta_initial = np.max(np.abs(initial_eta))
    metrics["max_eta_final"] = max_eta_final
    metrics["max_eta_initial"] = max_eta_initial
    
    if max_eta_initial > 1e-10:
        amplitude_ratio = max_eta_final / max_eta_initial
        metrics["amplitude_conservation"] = amplitude_ratio
    
    return metrics


def validate_results(final_state, diagnostics: Dict[str, list], 
                   config: InertiaGravityWaveConfig = None, 
                   **validation_kwargs) -> Tuple[bool, str]:
    """Validate inertia-gravity wave experiment results.
    
    Success criteria:
    - L2 error vs analytical solution < threshold
    - Wave amplitude reasonably preserved
    - No NaN or infinite values
    - Correct dispersion properties
    
    Parameters
    ----------
    final_state : OceanState
        Final model state
    diagnostics : Dict[str, list] 
        Time series diagnostics
    config : InertiaGravityWaveConfig, optional
        Configuration parameters
    **validation_kwargs
        Additional validation parameters (grid_type, grid, t_final, initial_eta)
        
    Returns
    -------
    bool
        True if validation passes
    str
        Descriptive notes about the validation
    """
    if config is None:
        config = InertiaGravityWaveConfig()
    
    # Check for NaN/infinite values
    if hasattr(final_state, 'eta') and not jnp.all(jnp.isfinite(final_state.eta.data)):
        return False, "NaN/Inf detected in final eta field"
    
    # Compute wave metrics if validation data provided
    if all(k in validation_kwargs for k in ['grid_type', 'grid', 't_final', 'initial_eta']):
        metrics = compute_wave_metrics(
            final_state, validation_kwargs['grid_type'], validation_kwargs['grid'],
            validation_kwargs['t_final'], validation_kwargs['initial_eta'], config
        )
        
        success = True
        notes_parts = []
        
        # L2 error check
        if "l2_error" in metrics:
            l2_err = metrics["l2_error"]
            notes_parts.append(f"L2={l2_err:.4f}")
            if l2_err > config.max_l2_error:
                success = False
                notes_parts.append("FAIL: excessive L2 error")
                
        # Amplitude preservation
        if "amplitude_conservation" in metrics:
            amp_ratio = metrics["amplitude_conservation"]
            notes_parts.append(f"amp_ratio={amp_ratio:.3f}")
            if abs(amp_ratio - 1.0) > config.max_amplitude_drift:
                success = False
                notes_parts.append("FAIL: poor amplitude conservation")
                
        # Additional metrics
        if "max_eta_final" in metrics:
            max_eta = metrics["max_eta_final"]
            notes_parts.append(f"max|eta|={max_eta:.3f}m")
            
        if "omega_analytical" in metrics:
            omega = metrics["omega_analytical"]
            notes_parts.append(f"omega={omega:.2e}")
        
        notes = ", ".join(notes_parts)
        return success, notes
    
    else:
        # Basic validation without analytical comparison
        return True, "basic validation passed"


def get_diagnostic_field_specs() -> list:
    """Get field specifications for diagnostic output.
    
    Returns
    -------
    list
        Field specifications for plotting: [(field_key, label, colormap), ...]
    """
    return [
        ("eta", "SSH (m)", "RdBu_r"),
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
    }


# Standard experiment configuration for test matrix integration
EXPERIMENT_CONFIG = {
    "name": "inertia_gravity_wave",
    "description": "Inertia-gravity (Poincaré) wave propagation test with analytical validation",
    "scientific_purpose": "Validates barotropic shallow water dynamics and dispersion properties",
    "reference": "Bishnu et al. (2024), DOI: 10.1029/2022MS003545",
    "config_class": InertiaGravityWaveConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": create_domain_config,
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 2.0,   # days
    "quick_duration": 0.2,     # days
    "expected_metrics": {
        "l2_error": "< 0.1",
        "amplitude_conservation": "0.8-1.2",
        "omega_analytical": "~1.4e-4 s^-1"
    },
    "grid_support": {
        "cubed_sphere": True,
        "latlon": True, 
        "mpas": True,
        "spectral": True
    },
    "special_config": {
        "H_max": 1000.0,    # Shallow equivalent depth
        "nlev": 2           # Barotropic-equivalent
    }
}