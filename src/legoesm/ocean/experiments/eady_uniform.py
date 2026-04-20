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
    lat_south: float = 16.0
    lat_north: float = 34.0
    lat_center: float = 25.0

    # Zonal extent (degrees). At 25°N, 1° ≈ 100.7 km, so 10° ≈ 1000 km.
    lon_west: float = 0.0
    lon_east: float = 10.0

    # Stratification: uniform N² via linear T(z)
    # N=1.2e-3 chosen so L_d=107km (11 grid cells at 10km),
    # λ_max≈428km (k=2 fits in 1000km domain).
    N: float = 1.2e-3
    T_ref: float = 10.0
    S_uniform: float = 35.0

    # Linear EOS
    alpha_T: float = 2.0e-4
    rho_0: float = 1025.0

    # Shear: U = Λz (zero at bottom, U_surface at top)
    # U_surface=0.8 gives τ≈5 days (classical Eady e-folding time).
    U_surface: float = 0.8

    # Jet envelope: Gaussian half-width in degrees. The jet and dT/dy
    # are localized around lat_center; the ocean outside the envelope
    # is a quiescent resting stratified state.
    jet_width_deg: float = 5.0

    # Depth scale for dT/dy: the meridional gradient decays as
    # exp(z / jet_depth_scale) so it is surface-intensified.
    # Set to H_max for depth-uniform (classical Eady, zero interior PV).
    jet_depth_scale: float = 5500.0

    # Perturbation: zonal wavenumber k seeds the most-unstable Eady
    # mode (l=0, k≠0). k=3 is the most unstable mode fitting in the
    # domain (μ=2.00, near μ_max=1.92). τ_undamped ≈ 5.7 days.
    T_perturbation_K: float = 0.1   # Large enough to dominate over discrete-balance noise
    perturbation_wavenumber: int = 3

    # Physics
    A_h: float = 0.0
    B_h: float = 1e10
    C_smag: float = 0.2
    K_h: float = 0.0                 # TVD advection handles grid-scale noise
    K_bih: float = 0.0
    A_v: float = 1.0e-5              # vertical viscosity [m^2/s]
    K_v: float = 5.0e-6              # vertical tracer diffusivity [m^2/s]
    bottom_drag_coeff: float = 0.001  # Linear bottom drag [m/s]; τ_bt≈64d, 4% of σ_Eady

    # Sponge layer: absorbs eddy energy near walls to prevent
    # Kelvin wave trapping and nonlinear steepening at boundaries.
    # Standard in MITgcm/MOM6 channel experiments (RBCS package).
    sponge_width_deg: float = 2.0
    sponge_timescale_days: float = 1.0

    barotropic_diffusion_alpha: float = 0.05
    barotropic_div_damp: float = 0.05

    @property
    def Lambda(self) -> float:
        """Effective shear scale using jet depth, not full ocean depth."""
        return self.U_surface / self.jet_depth_scale

    @property
    def dTdz(self) -> float:
        return self.N**2 / (g * self.alpha_T)

    @property
    def dTdy(self) -> float:
        """Thermal wind: f ∂u/∂z = -g α_T ∂T/∂y  →  ∂T/∂y = -f₀Λ/(gα_T) < 0."""
        f0 = 2.0 * Omega * np.sin(np.radians(self.lat_center))
        return -f0 * self.Lambda / (g * self.alpha_T)

    @property
    def f0(self) -> float:
        return 2.0 * Omega * np.sin(np.radians(self.lat_center))

    @property
    def Ld_km(self) -> float:
        """Deformation radius using jet depth scale."""
        return self.N * self.jet_depth_scale / self.f0 / 1e3

    @property
    def efolding_days(self) -> float:
        sigma = 0.31 * self.f0 * self.Lambda / self.N
        return 1.0 / sigma / 86400.0


def compute_sponge_mask(grid, config: EadyUniformConfig):
    """Compute sponge relaxation coefficient gamma(y) [1/s].

    Quadratic ramp from 0 in the interior to 1/tau at the walls.
    Works for both latlon (returns n_lat × n_lon) and MPAS (returns nCells).
    """
    tau = config.sponge_timescale_days * 86400.0
    south_edge = config.lat_south
    north_edge = config.lat_north
    w = config.sponge_width_deg

    if hasattr(grid, 'lat'):
        # Latlon grid
        lat_deg = np.degrees(np.asarray(grid.lat))
        gamma = np.zeros((grid.n_lat, grid.n_lon), dtype=np.float64)
        for i, lat in enumerate(lat_deg):
            dist_south = lat - south_edge
            dist_north = north_edge - lat
            if dist_south < w:
                gamma[i, :] = (1.0 - dist_south / w) ** 2 / tau
            elif dist_north < w:
                gamma[i, :] = (1.0 - dist_north / w) ** 2 / tau
    else:
        # MPAS grid
        lat_deg = np.degrees(np.asarray(grid.latCell))
        gamma = np.zeros(len(lat_deg), dtype=np.float64)
        for i, lat in enumerate(lat_deg):
            dist_south = lat - south_edge
            dist_north = north_edge - lat
            if dist_south < w:
                gamma[i] = (1.0 - dist_south / w) ** 2 / tau
            elif dist_north < w:
                gamma[i] = (1.0 - dist_north / w) ** 2 / tau

    return gamma


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
    """Set T = background stratification + depth-uniform meridional gradient.

    T(y,z) = T_ref + dTdz*z + dTdy * y_integrated_envelope(y)

    Classical Eady: dT/dy is depth-uniform, so N² is unaffected by the
    meridional gradient and the interior PV is zero.  The instability
    comes purely from the boundary temperature gradients (Eady 1949).

    Previous version used exp(z/D) weighting, which made dT/dy contribute
    to dT/dz and caused N² < 0 (convective instability) on the cold side
    of the jet when U_surface was large enough.
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

    # Depth-uniform dT/dy (classical Eady: no exp(z/D) weighting)
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
    """Set purely baroclinic velocity u = Λ(z + H/2) * envelope(y), eta = 0.

    The depth-mean is removed so the initial state is purely baroclinic
    with zero barotropic velocity and zero SSH.  This avoids the large
    (~1.5 m) geostrophic SSH signal from the depth-mean that would
    dominate over the perturbation and create confusing adjustment dynamics.

    The Eady instability operates on the baroclinic shear and boundary
    temperature gradients — it does not require a barotropic component.
    """
    z_full = np.asarray(z_coord.z_full_ref)
    dz = np.asarray(z_coord.dz_ref)
    nlev = len(z_full)

    # Purely baroclinic: remove depth-mean, don't put it into eta
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

    # eta = 0: no barotropic component, no geostrophic SSH needed
    return state._replace(
        u=Field(jnp.array(u_data), name="u",
                dims=state.u.dims, units=state.u.units))


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
    """Add zonal wavenumber T perturbation to seed the Eady mode (l=0, k≠0)."""
    if config.T_perturbation_K == 0:
        return state

    mask = np.asarray(state.land_mask.data)
    T_data = np.array(state.T.data, dtype=np.float64)

    lon_rad = np.asarray(grid.lon)
    lat_rad = np.asarray(grid.lat)
    lat_center_rad = np.radians(config.lat_center)
    lat_width_rad = np.radians(config.jet_width_deg)
    envelope = np.exp(-((lat_rad - lat_center_rad) / lat_width_rad) ** 2)

    k = config.perturbation_wavenumber
    # Fit k complete wavelengths in the periodic domain
    lon_west_rad = np.radians(config.lon_west)
    lon_east_rad = np.radians(config.lon_east)
    lon_frac = (lon_rad - lon_west_rad) / (lon_east_rad - lon_west_rad)
    zonal = np.sin(2.0 * np.pi * k * lon_frac)
    pert = config.T_perturbation_K * envelope[:, np.newaxis] * zonal[np.newaxis, :] * mask
    T_data[:, :, 0] += pert

    return state._replace(
        T=Field(jnp.array(T_data), name="T",
                dims=state.T.dims, units=state.T.units))


def _add_perturbation_mpas(state, mesh, config):
    """Add zonal wavenumber T perturbation to seed the Eady mode."""
    if config.T_perturbation_K == 0:
        return state

    lat_cell = np.asarray(mesh.latCell)
    lon_cell = np.asarray(mesh.lonCell)
    lat_center_rad = np.radians(config.lat_center)
    lat_width_rad = np.radians(config.jet_width_deg)
    envelope = np.exp(-((lat_cell - lat_center_rad) / lat_width_rad) ** 2)

    k = config.perturbation_wavenumber
    mask = np.asarray(state.land_mask.data)
    T_data = np.array(state.T.data, dtype=np.float64)

    # Fit k complete wavelengths in the periodic domain
    lon_west_rad = np.radians(config.lon_west)
    lon_east_rad = np.radians(config.lon_east)
    lon_frac = (lon_cell - lon_west_rad) / (lon_east_rad - lon_west_rad)
    pert = config.T_perturbation_K * np.sin(2.0 * np.pi * k * lon_frac) * envelope * mask
    T_data[:, 0] += pert

    return state._replace(
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

    # Bottom drag is applied via the lightweight config-level bottom_drag_r
    # (directly in the tendency function) to avoid the heavy physics pipeline
    # trace. All physics schemes set to "none" here.
    return OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
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
