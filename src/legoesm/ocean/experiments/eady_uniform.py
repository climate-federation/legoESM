"""Classical Eady Baroclinic Instability — Uniform Stratification.

A simplified Eady test with analytically tractable parameters:
- Uniform N² (linear temperature profile)
- Linear vertical shear U = Λz (zero at bottom)
- Uniform meridional temperature gradient dT/dy
- Linear equation of state
- η = 0 initial (barotropic mode damped via high diffusion)

The most-unstable Eady mode has:
  σ_max ≈ 0.31 × f₀ × Λ / N
  λ_max ≈ 4 × L_d  where L_d = NH/f₀

Default parameters (textbook regime at 2° resolution):
  N = 6.24e-3 s⁻¹, Λ = 9.09e-5 s⁻¹ (U_sfc = 0.5 m/s)
  L_d = 333 km, λ_max ≈ 1330 km (6 pts/wavelength at 2°)
  τ ≈ 25 days → 2.4 e-folds in 60 days

References:
- Eady (1949), "Long waves and cyclone waves"
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
class EadyUniformConfig:
    """Configuration for the classical Eady experiment."""
    # Domain
    H_max: float = 5500.0
    lat_south: float = 10.0
    lat_north: float = 80.0
    lat_center: float = 45.0

    # Stratification: uniform N² via linear T(z)
    N: float = 3e-3
    T_ref: float = 10.0
    S_uniform: float = 35.0

    # Linear EOS
    alpha_T: float = 2.0e-4
    rho_0: float = 1025.0

    # Shear: U = Λz (zero at bottom, U_surface at top)
    U_surface: float = 0.1

    # Jet envelope: Gaussian half-width in degrees. The jet and dT/dy
    # are localized around lat_center; the ocean outside the envelope
    # is a quiescent resting stratified state.
    jet_width_deg: float = 10.0

    # Perturbation
    eta_perturbation_m: float = 0.01
    T_perturbation_K: float = 0.1
    perturbation_wavenumber: int = 3

    # Physics — kept minimal so the instability can develop.
    # Classical Eady has no bottom drag or lateral diffusion.
    # Biharmonic viscosity damps the 2Δy grid mode without
    # affecting the physical Eady mode (~2600 km wavelength).
    A_h: float = 1e4
    B_h: float = 1e15
    bottom_drag_coeff: float = 0.0

    barotropic_diffusion_alpha: float = 0.005

    @property
    def Lambda(self) -> float:
        return self.U_surface / self.H_max

    @property
    def dTdz(self) -> float:
        return self.N**2 / (g * self.alpha_T)

    @property
    def dTdy(self) -> float:
        f0 = 2.0 * Omega * np.sin(np.radians(self.lat_center))
        return f0 * self.Lambda / (g * self.alpha_T)

    @property
    def f0(self) -> float:
        return 2.0 * Omega * np.sin(np.radians(self.lat_center))

    @property
    def Ld_km(self) -> float:
        return self.N * self.H_max / self.f0 / 1e3

    @property
    def efolding_days(self) -> float:
        sigma = 0.31 * self.f0 * self.Lambda / self.N
        return 1.0 / sigma / 86400.0


def create_initial_conditions(grid_type: str, grid, z_coord,
                              config: EadyUniformConfig = None):
    if config is None:
        config = EadyUniformConfig()

    if grid_type == "latlon_channel":
        state = _rest_state_latlon(grid, z_coord, config)
        state = _set_uniform_stratification(state, z_coord, config, grid)
        state = _set_linear_shear_latlon(state, grid, z_coord, config)
        state = _add_perturbation_latlon(state, grid, config)
        return state

    elif grid_type == "mpas_channel":
        state = _rest_state_mpas(grid, z_coord, config)
        state = _set_uniform_stratification_mpas(state, z_coord, config, grid)
        state = _set_linear_shear_mpas(state, grid, z_coord, config)
        state = _add_perturbation_mpas(state, grid, config)
        return state

    raise ValueError(f"Unsupported grid type: {grid_type}")


# ---------------------------------------------------------------------------
# Rest state construction
# ---------------------------------------------------------------------------

def _rest_state_latlon(grid, z_coord, config):
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    n_lat, n_lon = grid.n_lat, grid.n_lon
    wall_mask = np.ones((n_lat, n_lon), dtype=np.float32)
    wall_mask[0, :] = 0.0
    wall_mask[-1, :] = 0.0
    return rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=config.H_max,
        T_surface=config.T_ref, T_deep=config.T_ref,
        S_uniform=config.S_uniform,
        land_mask_override=wall_mask,
    )


def _rest_state_mpas(mesh, z_coord, config):
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    state = rest_state_mpas_ocean(
        mesh, z_coord, H_max=config.H_max,
        T_surface=config.T_ref, T_deep=config.T_ref,
        S_uniform=config.S_uniform,
    )
    lat_deg = np.degrees(np.asarray(mesh.latCell))
    ocean = (lat_deg >= config.lat_south) & (lat_deg <= config.lat_north)
    land_mask = np.where(ocean, 1.0, 0.0).astype(np.float32)
    return state._replace(
        land_mask=Field(jnp.array(land_mask), name="land_mask",
                        dims=state.land_mask.dims, units="1"))


# ---------------------------------------------------------------------------
# Uniform stratification + uniform dT/dy
# ---------------------------------------------------------------------------

def _jet_envelope(lat_deg, config):
    """Gaussian envelope centered on the jet: 1 at center, ~0 far away."""
    return np.exp(-((lat_deg - config.lat_center) / config.jet_width_deg) ** 2)


def _set_uniform_stratification(state, z_coord, config, grid):
    """Set T = background stratification + localized meridional gradient.

    T(y,z) = T_ref + dTdz*z + dTdy * y_integrated_envelope(y)

    The meridional gradient is localized by the jet envelope so that
    far from the jet center, T depends only on z (resting stratified state).
    """
    z_full = np.asarray(z_coord.z_full_ref)
    nlev = len(z_full)
    T_data = np.array(state.T.data, dtype=np.float64)
    n_lat, n_lon = T_data.shape[0], T_data.shape[1]

    lat_rad = np.asarray(grid.lat)
    lat_deg = np.degrees(lat_rad)
    lat_center_rad = np.radians(config.lat_center)
    R = 6.371e6
    y = (lat_rad - lat_center_rad) * R

    envelope = _jet_envelope(lat_deg, config)
    # Integrate envelope * dy to get the cumulative T anomaly:
    # T_anomaly(y) = dTdy * integral_0^y envelope(y') dy'
    T_anomaly = np.zeros(n_lat, dtype=np.float64)
    for i in range(1, n_lat):
        dy = y[i] - y[i - 1]
        T_anomaly[i] = T_anomaly[i - 1] + config.dTdy * 0.5 * (envelope[i] + envelope[i - 1]) * dy

    mask = np.asarray(state.land_mask.data)
    for k in range(nlev):
        T_zk = config.T_ref + config.dTdz * z_full[k]
        T_data[:, :, k] = (T_zk + T_anomaly[:, np.newaxis]) * mask

    return state._replace(T=Field(jnp.array(T_data), name="T",
                                  dims=state.T.dims, units=state.T.units))


def _set_uniform_stratification_mpas(state, z_coord, config, mesh):
    """Set T(y, z) for MPAS cells with localized jet envelope."""
    z_full = np.asarray(z_coord.z_full_ref)
    nlev = len(z_full)
    T_data = np.array(state.T.data, dtype=np.float64)

    lat_cell = np.asarray(mesh.latCell)
    lat_deg = np.degrees(lat_cell)
    lat_center_rad = np.radians(config.lat_center)
    R = 6.371e6
    y = (lat_cell - lat_center_rad) * R

    envelope = _jet_envelope(lat_deg, config)
    # For MPAS: approximate the integral by sorting cells by latitude
    sort_idx = np.argsort(lat_deg)
    y_sorted = y[sort_idx]
    env_sorted = envelope[sort_idx]
    T_anom_sorted = np.zeros(len(y), dtype=np.float64)
    for i in range(1, len(y)):
        dy = y_sorted[i] - y_sorted[i - 1]
        T_anom_sorted[i] = T_anom_sorted[i - 1] + config.dTdy * 0.5 * (env_sorted[i] + env_sorted[i - 1]) * dy
    T_anomaly = np.zeros(len(y), dtype=np.float64)
    T_anomaly[sort_idx] = T_anom_sorted

    mask = np.asarray(state.land_mask.data)
    for k in range(nlev):
        T_zk = config.T_ref + config.dTdz * z_full[k]
        T_data[:, k] = (T_zk + T_anomaly) * mask

    return state._replace(T=Field(jnp.array(T_data), name="T",
                                  dims=state.T.dims, units=state.T.units))


# ---------------------------------------------------------------------------
# Linear shear U = Λz
# ---------------------------------------------------------------------------

def _set_linear_shear_latlon(state, grid, z_coord, config):
    """Set u = Λz * envelope(y), remove depth mean, balance η."""
    z_full = np.asarray(z_coord.z_full_ref)
    dz = np.asarray(z_coord.dz_ref)
    nlev = len(z_full)

    U_profile = config.Lambda * z_full
    H_col = np.sum(dz)
    U_bar = np.sum(U_profile * dz) / H_col
    U_baroclinic = U_profile - U_bar

    lat_rad = np.asarray(grid.lat)
    lat_deg = np.degrees(lat_rad)
    envelope = _jet_envelope(lat_deg, config)

    u_data = np.zeros_like(state.u.data, dtype=np.float64)
    n_lat = u_data.shape[0]
    for i in range(n_lat):
        u_data[i, :, :] = U_baroclinic[np.newaxis, :] * envelope[i]

    mask = np.asarray(state.land_mask.data)
    if hasattr(state, 'u_mask') and state.u_mask is not None:
        u_mask = np.asarray(state.u_mask.data)
    else:
        u_mask = np.ones(u_data.shape[:2], dtype=np.float64)
    u_data *= u_mask[:, :, np.newaxis]

    # Geostrophic SSH: f0 * U_bar_envelope(y) = -g * deta/dy
    lat_center_rad = np.radians(config.lat_center)
    R = 6.371e6
    y = (lat_rad - lat_center_rad) * R
    U_bar_env = U_bar * envelope
    eta_1d = np.zeros(n_lat, dtype=np.float64)
    for i in range(1, n_lat):
        dy = y[i] - y[i - 1]
        eta_1d[i] = eta_1d[i - 1] - (config.f0 / g) * 0.5 * (U_bar_env[i] + U_bar_env[i - 1]) * dy
    ocean = mask > 0.5
    eta_1d_ocean = eta_1d[ocean[:, 0]] if ocean.any() else eta_1d
    eta_1d -= np.mean(eta_1d_ocean)
    eta_data = eta_1d[:, np.newaxis] * np.ones((1, grid.n_lon)) * mask

    return state._replace(
        u=Field(jnp.array(u_data), name="u",
                dims=state.u.dims, units=state.u.units),
        eta=Field(jnp.array(eta_data), name="eta",
                  dims=state.eta.dims, units=state.eta.units))


def _set_linear_shear_mpas(state, mesh, z_coord, config):
    """Set edge-normal velocity from zonal U = Λz * envelope(y), balance η."""
    z_full = np.asarray(z_coord.z_full_ref)
    dz = np.asarray(z_coord.dz_ref)
    nlev = len(z_full)

    U_profile = config.Lambda * z_full
    H_col = np.sum(dz)
    U_bar_val = np.sum(U_profile * dz) / H_col
    U_baroclinic = U_profile - U_bar_val

    angle = np.asarray(mesh.angleEdge)
    c1 = np.asarray(mesh.cellsOnEdge[0])
    c2 = np.asarray(mesh.cellsOnEdge[1])
    mask = np.asarray(state.land_mask.data)
    edge_mask = mask[c1] * mask[c2]

    lat_edge_deg = np.degrees(0.5 * (np.asarray(mesh.latCell)[c1]
                                      + np.asarray(mesh.latCell)[c2]))
    envelope_edge = _jet_envelope(lat_edge_deg, config)

    u_data = np.zeros_like(state.u.data, dtype=np.float64)
    for k in range(nlev):
        u_data[:, k] = U_baroclinic[k] * np.cos(angle) * edge_mask * envelope_edge

    # Geostrophic SSH
    lat_cell = np.asarray(mesh.latCell)
    lat_center_rad = np.radians(config.lat_center)
    R = 6.371e6
    y_offset = (lat_cell - lat_center_rad) * R
    eta_data = -(config.f0 / g) * U_bar_val * y_offset
    ocean = mask > 0.5
    if np.any(ocean):
        eta_data -= np.mean(eta_data[ocean])
    eta_data *= mask

    return state._replace(
        u=Field(jnp.array(u_data), name="u",
                dims=state.u.dims, units=state.u.units),
        eta=Field(jnp.array(eta_data), name="eta",
                  dims=state.eta.dims, units=state.eta.units))


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------

def _add_perturbation_latlon(state, grid, config):
    """Add SSH + temperature perturbation to seed the baroclinic Eady mode."""
    lon_rad = np.asarray(grid.lon)
    lat_rad = np.asarray(grid.lat)
    k = config.perturbation_wavenumber
    lon_2d, lat_2d = np.meshgrid(lon_rad, lat_rad)
    lat_center_rad = np.radians(config.lat_center)
    lat_width_rad = np.radians(10.0)
    envelope = np.exp(-((lat_2d - lat_center_rad) / lat_width_rad) ** 2)
    zonal = np.sin(k * lon_2d)

    eta_data = np.array(state.eta.data, dtype=np.float64)
    eta_data += config.eta_perturbation_m * zonal * envelope
    mask = np.asarray(state.land_mask.data)
    eta_data *= mask

    T_data = np.array(state.T.data, dtype=np.float64)
    nlev = T_data.shape[2]
    T_pert_2d = config.T_perturbation_K * zonal * envelope
    T_data[:, :, 0] += T_pert_2d * mask
    T_data[:, :, -1] -= T_pert_2d * mask

    return state._replace(
        eta=Field(jnp.array(eta_data), name="eta",
                  dims=state.eta.dims, units=state.eta.units),
        T=Field(jnp.array(T_data), name="T",
                dims=state.T.dims, units=state.T.units))


def _add_perturbation_mpas(state, mesh, config):
    """Add SSH + temperature perturbation to seed the baroclinic Eady mode."""
    lon_cell = np.asarray(mesh.lonCell)
    lat_cell = np.asarray(mesh.latCell)
    k = config.perturbation_wavenumber
    lat_center_rad = np.radians(config.lat_center)
    lat_width_rad = np.radians(10.0)
    envelope = np.exp(-((lat_cell - lat_center_rad) / lat_width_rad) ** 2)
    zonal = np.sin(k * lon_cell)

    eta_data = np.array(state.eta.data, dtype=np.float64)
    eta_data += config.eta_perturbation_m * zonal * envelope
    mask = np.asarray(state.land_mask.data)
    eta_data *= mask

    T_data = np.array(state.T.data, dtype=np.float64)
    T_pert_1d = config.T_perturbation_K * zonal * envelope
    T_data[:, 0] += T_pert_1d * mask
    T_data[:, -1] -= T_pert_1d * mask

    return state._replace(
        eta=Field(jnp.array(eta_data), name="eta",
                  dims=state.eta.dims, units=state.eta.units),
        T=Field(jnp.array(T_data), name="T",
                dims=state.T.dims, units=state.T.units))


# ---------------------------------------------------------------------------
# Physics / forcings
# ---------------------------------------------------------------------------

def create_forcings(grid_type: str, grid, config: EadyUniformConfig = None):
    if config is None:
        config = EadyUniformConfig()

    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig, ConstantVerticalMixingConfig,
    )
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.bottom_drag.config import (
        BottomDragConfig, LinearDragConfig,
    )
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig

    return OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        vertical_mixing=VerticalMixingConfig(
            scheme="constant",
            constant=ConstantVerticalMixingConfig(A_v=1e-3, K_v=1e-4),
        ),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(
            scheme="linear" if config.bottom_drag_coeff > 0 else "none",
            linear=LinearDragConfig(r=config.bottom_drag_coeff),
        ),
        convection=OceanConvectionConfig(scheme="none"),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_results(final_state, diagnostics: Dict[str, list],
                     config: EadyUniformConfig = None) -> Tuple[bool, str]:
    if config is None:
        config = EadyUniformConfig()

    for fname in ("u", "T", "S", "eta"):
        fdata = getattr(final_state, fname).data
        if not bool(jnp.all(jnp.isfinite(fdata))):
            return False, f"Non-finite values in {fname}"

    max_speed = diagnostics.get("max_speed", [0])[-1] if diagnostics.get("max_speed") else 0
    T_vals = diagnostics.get("mean_T", [])
    T_drift = abs(T_vals[-1] - T_vals[0]) if len(T_vals) >= 2 else 0

    notes = (f"max_speed={max_speed:.4f}m/s, T_drift={T_drift:.2e}, "
             f"Ld={config.Ld_km:.0f}km, tau={config.efolding_days:.0f}d")
    success = max_speed > 0.001 and T_drift < 5.0
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
    "name": "eady_uniform",
    "description": "Classical Eady instability: uniform N², linear shear, linear EOS",
    "scientific_purpose": (
        "Validates baroclinic instability growth rate against "
        "analytical Eady solution (σ ≈ 0.31 f₀ Λ/N)"
    ),
    "reference": "Eady (1949), Vallis (2017) Ch. 9",
    "config_class": EadyUniformConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": lambda config=None: {
        "H_max": (config or EadyUniformConfig()).H_max,
        "description": "Classical Eady: uniform N², linear shear, linear EOS",
    },
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 200.0,
    "quick_duration": 60.0,
    "expected_metrics": {
        "growth_rate": "σ ≈ 2.3e-7 s⁻¹ (exact Eady dispersion)",
        "efolding_time": "≈ 50 days",
        "most_unstable_wavelength": "≈ 2600 km",
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
