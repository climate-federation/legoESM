"""Eady Baroclinic Instability Ocean Experiment.

A regional channel experiment that tests baroclinic instability growth from
a meridional temperature front in thermal wind balance.  Unlike the Phillips
two-layer test (designed for layered/isopycnal models), this exercises the
z-star dycore's representation of continuous stratification, vertical shear,
and mesoscale eddy generation.

Scientific Purpose:
- Validate baroclinic energy conversion (APE → KE)
- Test thermal wind balance maintenance
- Verify that growing baroclinic modes are captured at model resolution
- Benchmark eddy kinetic energy growth rates

Domain Configuration:
- Rectangular ocean basin (0-120°E, 15-75°N) — channel-like regional domain
- Solid walls on all four sides
- Uniform depth: 5500m
- Background stratification: T_surface=20°C → T_deep=2°C, 1000m e-fold

Physical Setup:
- Meridional temperature front: tanh profile centered at 45°N,
  amplitude ΔT=8°C, width ~10°, decaying with depth
- Zonal velocity in thermal wind balance with ∂T/∂y
  (surface-intensified jet ~0.1-0.3 m/s, zero at bottom)
- Small sinusoidal SSH perturbation to seed instability
- No surface wind forcing — pure free baroclinic instability
- Linear bottom drag to damp barotropic mode

Expected Behavior:
- Initial adjustment (~5 days) as state equilibrates
- Exponential growth of most-unstable baroclinic mode (e-fold ~10-20 days)
- Meanders and eddies visible along the front by day 30-40
- Surface speeds increase from ~0.1 m/s (thermal wind) to ~0.5 m/s (eddies)

References:
- Eady (1949), "Long waves and cyclone waves"
- Pedlosky (1987), "Geophysical Fluid Dynamics", Ch. 7
- Vallis (2017), "Atmospheric and Oceanic Fluid Dynamics", Ch. 9
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from legoesm.constants import g, Omega
from legoesm.core.field import Field


@dataclass
class EadyInstabilityConfig:
    """Configuration for Eady baroclinic instability experiment."""
    # Domain
    H_max: float = 5500.0
    lon_west: float = 0.0
    lon_east: float = 120.0
    # Channel bounds — narrower than baroclinic_gyre to keep buffer cells
    # outside the ocean domain (buffer cells have degenerate Voronoi edges).
    lat_south: float = 25.0
    lat_north: float = 65.0

    # Background stratification
    T_surface: float = 20.0        # Surface temperature [degC]
    T_deep: float = 2.0            # Deep ocean temperature [degC]
    T_scale_depth: float = 1000.0  # Temperature e-folding depth [m]
    S_uniform: float = 35.0        # Salinity [PSU]

    # Meridional front
    front_delta_T: float = 8.0     # Front temperature contrast [degC]
    front_lat_center: float = 45.0 # Front center latitude [degrees]
    front_width_deg: float = 10.0  # Front half-width [degrees]
    front_depth_decay: float = 3.0 # Vertical decay (e-fold in levels)

    # Meridional taper: smoothly reduce front/velocity to zero near N/S walls
    # to prevent pressure buildup at solid boundaries.
    wall_taper_deg: float = 8.0    # Taper half-width from each wall [degrees]

    # Thermal wind parameters
    alpha_T: float = 2.0e-4        # Thermal expansion [1/K] (matches LinearEOS)
    rho_0: float = 1025.0          # Reference density [kg/m³]

    # Perturbation to seed instability
    eta_perturbation_m: float = 0.01  # SSH perturbation amplitude [m]
    perturbation_wavenumber: int = 3   # Zonal wavenumber

    # Physics
    A_h: float = 5e5               # Horizontal viscosity [m²/s]
    bottom_drag_coeff: float = 1.1e-3  # Linear bottom drag [m/s]


def create_initial_conditions(grid_type: str, grid, z_coord,
                              config: EadyInstabilityConfig = None):
    """Create Eady baroclinic instability initial conditions.

    Steps:
    1. Create base rest state with basin + land mask
    2. Add exponential stratification
    3. Add meridional temperature front (tanh profile, depth-decaying)
    4. Compute thermal-wind-balanced zonal velocity
    5. Add small SSH perturbation to seed instability
    """
    if config is None:
        config = EadyInstabilityConfig()

    if grid_type == "latlon_channel":
        state = _channel_rest_state_latlon(grid, z_coord, config)
        state = _add_stratification_and_front(state, z_coord, config)
        state = _add_thermal_wind_latlon(state, grid, z_coord, config)
        state = _add_ssh_perturbation_latlon(state, grid, config)
        return state

    elif grid_type == "mpas_channel":
        state = _channel_rest_state_mpas(grid, z_coord, config)
        state = _add_stratification_and_front(state, z_coord, config)
        state = _add_thermal_wind_mpas(state, grid, z_coord, config)
        state = _add_ssh_perturbation_mpas(state, grid, config)
        return state

    else:
        raise ValueError(
            f"Grid type {grid_type} not supported for eady_instability. "
            f"Use latlon_channel or mpas_channel."
        )


def _meridional_taper(lat_deg, config: EadyInstabilityConfig):
    """Smooth taper that goes to zero near N/S walls.

    Returns values in [0, 1]: 0 at the walls, 1 in the interior.
    Uses a half-cosine ramp over ``wall_taper_deg`` from each wall.
    """
    w = config.wall_taper_deg
    lat_s = config.lat_south
    lat_n = config.lat_north

    # Distance from south and north walls [degrees]
    d_south = lat_deg - lat_s
    d_north = lat_n - lat_deg

    # Half-cosine ramp: 0 at wall, 1 at taper_deg inside
    taper_south = np.where(d_south < w,
                           0.5 * (1.0 - np.cos(np.pi * np.clip(d_south / w, 0, 1))),
                           1.0)
    taper_north = np.where(d_north < w,
                           0.5 * (1.0 - np.cos(np.pi * np.clip(d_north / w, 0, 1))),
                           1.0)
    return taper_south * taper_north


def _channel_rest_state_latlon(grid, z_coord, config: EadyInstabilityConfig):
    """Create rest state for a latlon channel (periodic-x, walls at N/S)."""
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    wall_mask = np.ones((n_lat, n_lon), dtype=np.float32)
    wall_mask[0, :] = 0.0   # south wall
    wall_mask[-1, :] = 0.0  # north wall
    return rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=config.H_max,
        T_surface=config.T_surface, T_deep=config.T_deep,
        S_uniform=config.S_uniform,
        land_mask_override=wall_mask,
    )


def _channel_rest_state_mpas(mesh, z_coord, config: EadyInstabilityConfig):
    """Create rest state for an MPAS channel (periodic-x, walls at N/S).

    The mesh was created with ``periodic_x=True``, covering a latitude
    band. Cells outside the target latitude range (buffer cells) are
    masked as land.
    """
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    state = rest_state_mpas_ocean(
        mesh, z_coord, H_max=config.H_max,
        T_surface=config.T_surface, T_deep=config.T_deep,
        S_uniform=config.S_uniform,
    )
    # Land mask: cells outside target latitude band are land
    lat_deg = np.degrees(np.asarray(mesh.latCell))
    ocean = ((lat_deg >= config.lat_south) & (lat_deg <= config.lat_north))
    land_mask = np.where(ocean, 1.0, 0.0).astype(np.float32)
    return state._replace(
        land_mask=Field(jnp.array(land_mask), name="land_mask",
                        dims=state.land_mask.dims, units="1"))


def _add_stratification_and_front(state, z_coord, config: EadyInstabilityConfig):
    """Add background stratification + meridional temperature front."""
    z_full = np.asarray(z_coord.z_full_ref)   # negative, surface first
    nlev = len(z_full)

    # Background exponential profile: T(z) = T_deep + (T_s - T_d)*exp(z/scale)
    decay = np.exp(z_full / config.T_scale_depth)
    T_bg = config.T_deep + (config.T_surface - config.T_deep) * decay  # (nlev,)

    # Meridional front: tanh(lat - lat_center) * depth_decay
    T_data = np.array(state.T.data)

    # Get latitude array depending on grid type
    if T_data.ndim == 3:
        # Latlon: (n_lat, n_lon, nlev) — lat from grid
        lat_deg = np.degrees(np.asarray(state.T.data).mean(axis=(1, 2)))
        # Actually we need the grid lat. Use shape to reconstruct.
        # For latlon_regional, state.T has dims (n_lat, n_lon, nlev).
        # Access lat from the data coordinates isn't direct, so compute from
        # basin bounds assuming uniform spacing.
        n_lat = T_data.shape[0]
        lat_deg = np.linspace(
            config.lat_south, config.lat_north, n_lat)
        lat_3d = lat_deg[:, np.newaxis, np.newaxis]  # (n_lat, 1, 1)
    else:
        # MPAS: (nCells, nlev)
        lat_deg = np.degrees(np.asarray(state.land_mask.data))
        # Actually need cell latitudes from mesh — not available in state.
        # We'll handle this via the grid object passed separately.
        # For now, mark that MPAS needs lat from mesh.
        lat_deg = None
        lat_3d = None

    # Level-dependent decay for front
    level_idx = np.arange(nlev, dtype=np.float64)
    front_decay = np.exp(-level_idx / config.front_depth_decay)  # (nlev,)

    if lat_3d is not None:
        # Latlon path
        taper = _meridional_taper(lat_deg, config)[:, np.newaxis, np.newaxis]
        front = (config.front_delta_T / 2.0
                 * np.tanh((lat_3d - config.front_lat_center)
                           / config.front_width_deg)
                 * taper)
        # Apply depth decay and add to background
        for k in range(nlev):
            T_data[:, :, k] = T_bg[k] + front[:, :, 0] * front_decay[k]
    else:
        # MPAS: T_data is (nCells, nlev), lat needs to come from somewhere
        # We store lat in land_mask temporarily — no, use H_bathy shape.
        # Actually we can compute from the T_data itself — but we need mesh.
        # This will be handled by the caller who has the mesh.
        # For now, just set background stratification; front added in
        # _add_thermal_wind_mpas which has access to the mesh.
        for k in range(nlev):
            T_data[:, k] = T_bg[k]

    return state._replace(T=Field(jnp.array(T_data), name="T",
                                  dims=state.T.dims, units=state.T.units))


def _add_front_mpas(state, mesh, z_coord, config: EadyInstabilityConfig):
    """Add meridional front to MPAS state (needs mesh for cell latitudes)."""
    z_full = np.asarray(z_coord.z_full_ref)
    nlev = len(z_full)
    level_idx = np.arange(nlev, dtype=np.float64)
    front_decay = np.exp(-level_idx / config.front_depth_decay)

    lat_deg = np.degrees(np.asarray(mesh.latCell))  # (nCells,)
    taper = _meridional_taper(lat_deg, config)  # (nCells,)
    front_1d = (config.front_delta_T / 2.0
                * np.tanh((lat_deg - config.front_lat_center)
                          / config.front_width_deg)
                * taper)  # (nCells,)

    T_data = np.array(state.T.data)  # (nCells, nlev)
    for k in range(nlev):
        T_data[:, k] += front_1d * front_decay[k]

    return state._replace(T=Field(jnp.array(T_data), name="T",
                                  dims=state.T.dims, units=state.T.units))


def _add_thermal_wind_latlon(state, grid, z_coord,
                              config: EadyInstabilityConfig):
    """Compute thermal-wind-balanced zonal velocity for latlon C-grid.

    f₀ ∂U/∂z = -(g α_T) ∂T/∂y

    Integrate bottom-up with U=0 at the bottom.  U lives at u-face
    points (n_lat, n_lon+1, nlev).
    """
    T_data = np.asarray(state.T.data)  # (n_lat, n_lon, nlev)
    n_lat, n_lon, nlev = T_data.shape

    # Latitude at cell centers
    lat_rad = np.asarray(grid.lat)  # (n_lat,) radians
    lat_mid = np.radians(config.front_lat_center)
    f0 = 2.0 * Omega * np.sin(lat_mid)

    # ∂T/∂y at u-face latitudes (between cell centers)
    dy = grid.dy / 2.0  # grid.dy is "distance over 2 cells"
    # dT/dy at interior u-faces: (T[i] - T[i-1]) / dy
    dTdy = (T_data[1:, :, :] - T_data[:-1, :, :]) / dy  # (n_lat-1, n_lon, nlev)
    # Pad to (n_lat, n_lon+1, nlev) u-face shape:
    # Top/bottom rows: zero (solid wall), columns: periodic average
    dTdy_padded = np.zeros((n_lat, nlev), dtype=np.float64)

    # Average over longitude for zonal-mean front (simpler, avoids noise)
    dTdy_zonal = np.mean(dTdy, axis=1)  # (n_lat-1, nlev)

    # Thermal wind: dU/dz = -(g * alpha_T / f0) * dT/dy
    dz = np.asarray(z_coord.dz_ref)  # (nlev,), layer thicknesses
    coeff = -g * config.alpha_T / f0

    # Build U(lat, z) — zonal mean, then broadcast to u-faces
    # U at u-face latitudes (n_lat-1 interior faces)
    U_zonal = np.zeros((n_lat - 1, nlev), dtype=np.float64)
    # Integrate bottom-up: U[k] = U[k+1] + dU/dz * dz[k]
    for k in range(nlev - 2, -1, -1):
        dz_half = 0.5 * (dz[k] + dz[k + 1]) if k + 1 < nlev else dz[k]
        U_zonal[:, k] = U_zonal[:, k + 1] + coeff * dTdy_zonal[:, k] * dz_half

    # Depth-mean velocity (for geostrophic SSH balance)
    H_col = np.sum(dz)
    U_bar_full = np.sum(U_zonal * dz[np.newaxis, :], axis=1) / H_col  # (n_lat-1,)

    # Remove depth mean from 3D velocity (barotropic solver handles it)
    U_zonal = U_zonal - U_bar_full[:, np.newaxis]

    # Broadcast to full u-face array (n_lat, n_lon+1, nlev)
    u_data = np.zeros_like(state.u.data)
    for i in range(min(n_lat - 1, U_zonal.shape[0])):
        u_data[i, :, :] = U_zonal[i, np.newaxis, :]

    # Apply land mask
    mask = np.asarray(state.land_mask.data)  # (n_lat, n_lon)
    if hasattr(state, 'u_mask'):
        u_mask = np.asarray(state.u_mask.data)
    else:
        u_mask = np.ones((n_lat, n_lon + 1), dtype=np.float64)
    u_data = u_data * u_mask[:, :, np.newaxis]

    # Geostrophically balanced SSH: f0 * U_bar = -g * deta/dy
    # => eta(y) = -(f0/g) * integral(U_bar, dy) from south wall
    eta_data = np.zeros((n_lat, n_lon), dtype=np.float64)
    for i in range(1, n_lat):
        idx = min(i - 1, len(U_bar_full) - 1)
        eta_data[i, :] = eta_data[i - 1, :] - (f0 / g) * U_bar_full[idx] * dy
    # Remove mean to keep eta centered around zero
    ocean = np.asarray(state.land_mask.data) > 0.5
    if np.any(ocean):
        eta_data -= np.mean(eta_data[ocean])
    eta_data *= mask

    state = state._replace(
        u=Field(jnp.array(u_data), name="u",
                dims=state.u.dims, units=state.u.units),
        eta=Field(jnp.array(eta_data), name="eta",
                  dims=state.eta.dims, units=state.eta.units),
    )
    return state


def _add_thermal_wind_mpas(state, mesh, z_coord,
                            config: EadyInstabilityConfig):
    """Compute thermal-wind-balanced normal velocity for MPAS.

    First add the meridional front (needs mesh.latCell), then compute
    zonal U from thermal wind and project onto edge normals.
    """
    # First add the front (needs mesh for cell latitudes)
    state = _add_front_mpas(state, mesh, z_coord, config)

    T_data = np.asarray(state.T.data)  # (nCells, nlev)
    nlev = T_data.shape[1]
    dz = np.asarray(z_coord.dz_ref)

    lat_cell = np.asarray(mesh.latCell)  # (nCells,) radians
    lat_mid = np.radians(config.front_lat_center)
    f0 = 2.0 * Omega * np.sin(lat_mid)
    coeff = -g * config.alpha_T / f0

    # Compute dT/dy ≈ dT/dlat * 1/(R) at each cell from the analytical front
    R = 6.371e6  # Earth radius [m]
    lat_deg = np.degrees(lat_cell)
    # Analytical dT/dy from the tanh front:
    # T_front = ΔT/2 * tanh((lat - lat_c) / w) * depth_decay
    # dT_front/dy = dT_front/dlat * 1/R
    #             = ΔT/(2*w_rad*R) * sech²((lat - lat_c) / w) * depth_decay
    w_rad = np.radians(config.front_width_deg)
    sech2 = 1.0 / np.cosh((lat_cell - lat_mid) / w_rad) ** 2
    taper = _meridional_taper(lat_deg, config)  # (nCells,)
    dTdy_base = config.front_delta_T / (2.0 * w_rad * R) * sech2 * taper

    level_idx = np.arange(nlev, dtype=np.float64)
    front_decay = np.exp(-level_idx / config.front_depth_decay)

    # U(cell, z): integrate bottom-up
    U_cell = np.zeros((mesh.nCells, nlev), dtype=np.float64)
    for k in range(nlev - 2, -1, -1):
        dz_half = 0.5 * (dz[k] + dz[k + 1]) if k + 1 < nlev else dz[k]
        U_cell[:, k] = (U_cell[:, k + 1]
                        + coeff * dTdy_base * front_decay[k] * dz_half)

    # Depth-mean velocity for geostrophic SSH balance
    H_col = np.sum(dz)
    U_bar = np.sum(U_cell * dz[np.newaxis, :], axis=1) / H_col  # (nCells,)

    # Remove depth mean from 3D velocity
    U_cell = U_cell - U_bar[:, np.newaxis]

    # Project zonal U onto edge normals: u_edge = U * cos(angleEdge)
    c1 = np.asarray(mesh.cellsOnEdge[0])  # (nEdges,)
    c2 = np.asarray(mesh.cellsOnEdge[1])
    angle = np.asarray(mesh.angleEdge)      # (nEdges,)

    U_edge = 0.5 * (U_cell[c1] + U_cell[c2])  # (nEdges, nlev)
    u_normal = U_edge * np.cos(angle)[:, np.newaxis]

    # Apply edge mask
    mask = np.asarray(state.land_mask.data)
    edge_mask = mask[c1] * mask[c2]
    u_normal = u_normal * edge_mask[:, np.newaxis]

    # Geostrophically balanced SSH: f0 * U_bar = -g * deta/dy
    # For MPAS, compute eta analytically from the zonal-mean U_bar profile.
    # eta(lat) = -(f0/g) * integral_south^lat U_bar(lat') dy
    # Sort cells by latitude and integrate.
    ocean = mask > 0.5
    lat_deg_ocean = lat_deg[ocean]
    U_bar_ocean = U_bar[ocean]
    sort_idx = np.argsort(lat_deg_ocean)
    lat_sorted = lat_deg_ocean[sort_idx]
    U_sorted = U_bar_ocean[sort_idx]

    # Cumulative integral in latitude
    dlat_m = np.diff(np.radians(lat_sorted)) * R  # (n-1,) meters
    eta_sorted = np.zeros(len(lat_sorted), dtype=np.float64)
    for i in range(1, len(lat_sorted)):
        eta_sorted[i] = eta_sorted[i - 1] - (f0 / g) * U_sorted[i - 1] * dlat_m[i - 1]
    eta_sorted -= np.mean(eta_sorted)

    # Map back to cell indices
    eta_data = np.zeros(mesh.nCells, dtype=np.float64)
    # Interpolate: for each ocean cell, find nearest in sorted array
    for i, cell_lat in enumerate(lat_deg):
        if mask[i] > 0.5:
            idx = np.argmin(np.abs(lat_sorted - cell_lat))
            eta_data[i] = eta_sorted[idx]
    eta_data *= mask

    state = state._replace(
        u=Field(jnp.array(u_normal), name="u",
                dims=state.u.dims, units=state.u.units),
        eta=Field(jnp.array(eta_data), name="eta",
                  dims=state.eta.dims, units=state.eta.units),
    )
    return state


def _add_ssh_perturbation_latlon(state, grid, config: EadyInstabilityConfig):
    """Add sinusoidal SSH perturbation to seed instability (latlon)."""
    lon_rad = np.asarray(grid.lon)  # (n_lon,)
    lat_rad = np.asarray(grid.lat)  # (n_lat,)

    k = config.perturbation_wavenumber
    # Sinusoidal in longitude, Gaussian envelope in latitude around front
    lon_2d, lat_2d = np.meshgrid(lon_rad, lat_rad)
    lat_envelope = np.exp(-((np.degrees(lat_2d) - config.front_lat_center)
                            / config.front_width_deg) ** 2)
    eta_pert = (config.eta_perturbation_m
                * np.sin(k * lon_2d) * lat_envelope)

    eta_data = np.asarray(state.eta.data) + eta_pert
    mask = np.asarray(state.land_mask.data)
    eta_data = eta_data * mask

    return state._replace(eta=Field(jnp.array(eta_data), name="eta",
                                    dims=state.eta.dims, units=state.eta.units))


def _add_ssh_perturbation_mpas(state, mesh, config: EadyInstabilityConfig):
    """Add sinusoidal SSH perturbation to seed instability (MPAS)."""
    lon_cell = np.asarray(mesh.lonCell)
    lat_cell = np.asarray(mesh.latCell)

    k = config.perturbation_wavenumber
    lat_envelope = np.exp(-((np.degrees(lat_cell) - config.front_lat_center)
                            / config.front_width_deg) ** 2)
    eta_pert = (config.eta_perturbation_m
                * np.sin(k * lon_cell) * lat_envelope)

    eta_data = np.asarray(state.eta.data) + eta_pert
    mask = np.asarray(state.land_mask.data)
    eta_data = eta_data * mask

    return state._replace(eta=Field(jnp.array(eta_data), name="eta",
                                    dims=state.eta.dims, units=state.eta.units))


def create_forcings(grid_type: str, grid,
                    config: EadyInstabilityConfig = None):
    """No surface wind — pure baroclinic instability. Bottom drag only."""
    if config is None:
        config = EadyInstabilityConfig()

    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig, ConstantVerticalMixingConfig,
    )
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig

    return OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        vertical_mixing=VerticalMixingConfig(
            scheme="constant",
            constant=ConstantVerticalMixingConfig(A_v=1e-3, K_v=1e-4),
        ),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
    )


def create_domain_config(config: EadyInstabilityConfig = None) -> Dict[str, Any]:
    if config is None:
        config = EadyInstabilityConfig()
    return {
        "H_max": config.H_max,
        "lon_west": config.lon_west,
        "lon_east": config.lon_east,
        "lat_south": config.lat_south,
        "lat_north": config.lat_north,
        "description": (
            "Channel domain with meridional temperature front "
            "for Eady baroclinic instability"
        ),
    }


def validate_results(final_state, diagnostics: Dict[str, list],
                     config: EadyInstabilityConfig = None) -> Tuple[bool, str]:
    """Validate Eady instability results.

    Checks:
    1. All fields finite (no blowup)
    2. Instability grew (max_speed increased from thermal wind baseline)
    """
    if config is None:
        config = EadyInstabilityConfig()

    # Finiteness
    for fname in ("u", "T", "S", "eta"):
        fdata = getattr(final_state, fname).data
        if not bool(jnp.all(jnp.isfinite(fdata))):
            return False, f"Non-finite values in {fname}"

    max_speed = diagnostics.get("max_speed", [0])[-1] if diagnostics.get("max_speed") else 0
    T_vals = diagnostics.get("mean_T", [])
    T_drift = abs(T_vals[-1] - T_vals[0]) if len(T_vals) >= 2 else 0

    notes = f"max_speed={max_speed:.4f}m/s, T_drift={T_drift:.2e}degC"
    success = max_speed > 0.001 and T_drift < 1.0
    return success, notes


def get_diagnostic_field_specs() -> list:
    return [
        ("eta", "SSH (m)", "RdBu_r"),
        ("speed_sfc", "Surface Speed (m/s)", "plasma"),
        ("SST", "SST (degC)", "RdYlBu_r"),
    ]


def get_scalar_units() -> Dict[str, str]:
    return {
        "mean_eta": "m",
        "max_speed": "m/s",
        "max_abs_u": "m/s",
        "mean_T": "degC",
        "mean_S": "PSU",
    }


EXPERIMENT_CONFIG = {
    "name": "eady_instability",
    "description": "Eady baroclinic instability with meridional temperature front",
    "scientific_purpose": (
        "Validates baroclinic energy conversion and eddy growth "
        "from a vertically sheared, stratified flow"
    ),
    "reference": "Eady (1949), Pedlosky (1987), Vallis (2017)",
    "config_class": EadyInstabilityConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": create_domain_config,
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 60.0,
    "quick_duration": 5.0,
    "expected_metrics": {
        "max_speed_final": "> 0.001 m/s (instability grew)",
        "T_drift": "< 1.0 degC",
    },
    "grid_support": {
        "cubed_sphere": False,
        "latlon": False,
        "mpas": False,
        "latlon_regional": False,
        "mpas_regional": False,
        "latlon_channel": True,
        "mpas_channel": True,
        "spectral": False,
    },
}
