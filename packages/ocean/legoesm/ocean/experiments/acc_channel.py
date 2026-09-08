"""ACC-like Channel with Gaussian Ridge.

Wind-driven stratified channel on the sphere with a meridional Gaussian
ridge, inspired by Zhang et al. (2024, JPO).  The experiment exercises
the z-star coordinate's handling of spatially varying bathymetry --
a capability not tested by any other ocean test matrix case.

Scientific Purpose:
- Validate flow over variable bathymetry in z-star coordinates
- Test topographic form stress generation by a submarine ridge
- Benchmark wind-driven zonal transport in a re-entrant channel
- Exercise the sponge layer (northern boundary restoring)
- Stress-test the Jacobian with a 1000 m ridge in 3000 m depth

Domain Configuration:
- Zonally periodic channel on the sphere (~40S)
- Meridional walls at north and south (free-slip)
- Gaussian ridge running meridionally (wall-to-wall, no taper)
- H_max = 3000 m, ridge crest at 2000 m

Physical Setup:
- Exponential stratification (Abernathey et al. 2011 profile)
- Linear EOS (temperature only, salinity passive)
- Half-sine zonal wind stress peaking at channel center
- Northern boundary sponge restoring T toward initial profile
- Linear bottom drag

Expected Behavior:
- Wind-driven zonal flow develops within days
- Topographic form stress balances wind stress within ~1 month
- Flow accelerates/decelerates over the ridge
- Sponge maintains thermocline at northern boundary

References:
- Zhang, Nikurashin, Pena-Molino, Rintoul, Doddridge (2024),
  J. Phys. Oceanogr., 54, 1565-1581. DOI: 10.1175/JPO-D-23-0042.1
- Abernathey, Marshall, Ferreira (2011), J. Phys. Oceanogr., 41, 2264-2284.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass

from legoesm import constants
from legoesm.core.field import Field


@dataclass
class ACCChannelConfig:
    """Configuration for the ACC-like channel experiment."""

    # Domain (spherical channel centered at ~40S)
    lat_south: float = -50.0
    lat_north: float = -30.0
    lon_west: float = 0.0
    lon_east: float = 18.0   # ~1500 km at 40S (for latlon_channel)

    # Depth
    H_max: float = 3000.0

    # Gaussian ridge (meridional, varying in longitude)
    ridge_height: float = 1000.0     # m, peak height above abyssal plain
    ridge_sigma_km: float = 150.0    # km, Gaussian e-folding width
    ridge_lon_center: float | None = None  # degrees; None = domain center

    # Stratification: Abernathey et al. (2011) exponential profile
    # Tstar(z) = delta_T * (exp(z/h) - exp(-H/h)) / (1 - exp(-H/h))
    delta_T: float = 8.0            # degC, surface-to-bottom range
    thermocline_scale: float = 1000.0  # m, e-folding depth
    S_uniform: float = 35.0         # PSU (passive, not stepped)

    # Linear EOS
    alpha_T: float = 2.0e-4
    rho_0: float = constants.rho_ocean
    T_ref_C: float = 4.0              # reference T for EOS [degC]

    # Wind forcing
    tau0: float = 0.1               # N/m^2, peak zonal stress

    # Bottom drag (note: current legoESM uses [1/s]; #202 will fix to [m/s])
    bottom_drag_coeff: float = 1.1e-3

    # Sponge (northern boundary only)
    sponge_width_deg: float = 2.0   # ~200 km at 40S
    sponge_timescale_days: float = 7.0

    # Perturbation to break symmetry
    T_perturbation_K: float = 1e-3

    # Physics / dissipation
    # MOM6-inspired: biharmonic Smagorinsky only, no Laplacian viscosity
    # or tracer diffusion.  C_smag^2 ≈ MOM6 SMAG_BI_CONST (0.06).
    # Vertical mixing matches MOM6 background values (KV=1e-5, KD=5e-6).
    A_h: float = 0.0                # Laplacian viscosity [m^2/s]
    B_h: float = 0.0                # biharmonic viscosity (constant)
    C_smag: float = 0.25            # Smagorinsky coeff; C_smag^2 ≈ 0.06
    K_h: float = 0.0                # tracer diffusivity [m^2/s]

    # Vertical mixing (explicit solver)
    A_v: float = 1.0e-5             # vertical viscosity [m^2/s]
    K_v: float = 5.0e-6             # vertical tracer diffusivity [m^2/s]

    # Free-surface Laplacian: OFF. Inherited from the collocated/cubed-sphere
    # solvers (0.05 is the CUBED-SPHERE default, raised there for a
    # face-boundary feedback); this is a C-grid case, which has no such
    # checkerboard mode. At its resolution the setting implied a diffusivity
    # of order 1e6 m^2/s against 1e2-1e3 for the real ocean, and it moved far
    # more water than the flow itself.
    #
    # Measured, one variable at a time: with it OFF this case PASSES on both
    # its grids, and it also passes with the derived velocity viscosity added,
    # so no replacement is required -- unlike eady_uniform, which needed one.
    barotropic_diffusion_alpha: float = 0.0
    barotropic_div_damp: float = 0.05

    @property
    def ridge_lon_center_deg(self) -> float:
        if self.ridge_lon_center is not None:
            return self.ridge_lon_center
        return 0.5 * (self.lon_west + self.lon_east)


# ---------------------------------------------------------------------------
# Stratification profile
# ---------------------------------------------------------------------------

def _abernathey_profile(z, delta_T, h, H_max):
    """Abernathey et al. (2011) Eq. (2) exponential thermocline.

    Returns T(z) where z is negative (below surface).
    T ~ delta_T at surface, ~ 0 at bottom.
    """
    return delta_T * (np.exp(z / h) - np.exp(-H_max / h)) / (
        1.0 - np.exp(-H_max / h))


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def create_initial_conditions(grid_type: str, grid, z_coord,
                              config: ACCChannelConfig = None):
    """Create ACC channel initial conditions for latlon or MPAS channel grids.

    Steps:
    1. Create rest state with wall-masked boundaries
    2. Set exponential stratification (Abernathey profile)
    3. Set Gaussian ridge bathymetry
    4. Add small temperature perturbation
    """
    if config is None:
        config = ACCChannelConfig()

    if grid_type == "latlon_channel":
        return _init_latlon(grid, z_coord, config)
    elif grid_type == "mpas_channel":
        return _init_mpas(grid, z_coord, config)
    raise ValueError(f"Unsupported grid type for ACC channel: {grid_type}")


def _init_latlon(grid, z_coord, config):
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean

    n_lat, n_lon = grid.n_lat, grid.n_lon

    # Wall mask: land at north/south boundaries
    wall_mask = np.ones((n_lat, n_lon), dtype=np.float32)
    wall_mask[0, :] = 0.0
    wall_mask[-1, :] = 0.0

    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=config.H_max,
        T_water_init_C=config.delta_T, T_deep=0.0,
        S_uniform=config.S_uniform,
        land_mask_override=wall_mask,
    )

    # --- Stratification: Abernathey profile ---
    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    T_profile = _abernathey_profile(
        z_full, config.delta_T, config.thermocline_scale, config.H_max)

    mask = np.asarray(state.land_mask.data, dtype=np.float64)
    nlev = z_coord.n_levels
    T_data = np.zeros((n_lat, n_lon, nlev), dtype=np.float64)
    for k in range(nlev):
        T_data[..., k] = T_profile[k] * mask

    # Add small perturbation to break symmetry
    rng = np.random.default_rng(seed=42)
    T_data += config.T_perturbation_K * rng.standard_normal(T_data.shape) * mask[..., np.newaxis]

    state = state._replace(
        T=Field(jnp.array(T_data), name="T",
                dims=state.T.dims, units="degC"))

    # --- Gaussian ridge bathymetry ---
    lon_deg = np.degrees(np.asarray(grid.lon, dtype=np.float64))  # (n_lat, n_lon)
    lat_center_rad = np.radians(0.5 * (config.lat_south + config.lat_north))
    # Convert sigma from km to degrees at channel center latitude
    km_per_deg = 111.32 * np.cos(lat_center_rad)
    sigma_deg = config.ridge_sigma_km / km_per_deg
    lon_c = config.ridge_lon_center_deg

    H_bathy = config.H_max - config.ridge_height * np.exp(
        -((lon_deg - lon_c) ** 2) / (sigma_deg ** 2))

    # Keep land cells at H_max (smooth Jacobian), enforce minimum depth
    H_bathy = np.where(mask > 0.5, np.maximum(H_bathy, 50.0), config.H_max)

    state = state._replace(
        H_bathy=Field(jnp.array(H_bathy), name="H_bathy",
                      dims=state.H_bathy.dims, units="m"))

    return state


def _init_mpas(mesh, z_coord, config):
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean

    state = rest_state_mpas_ocean(
        mesh, z_coord, H_max=config.H_max,
        T_water_init_C=config.delta_T, T_deep=0.0,
        S_uniform=config.S_uniform,
    )

    # Set land mask from latitude bounds
    lat_deg = np.degrees(np.asarray(mesh.latCell, dtype=np.float64))
    ocean = (lat_deg >= config.lat_south) & (lat_deg <= config.lat_north)
    land_mask = np.where(ocean, 1.0, 0.0).astype(np.float32)
    state = state._replace(
        land_mask=Field(jnp.array(land_mask), name="land_mask",
                        dims=state.land_mask.dims, units="1"))

    # --- Stratification ---
    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    T_profile = _abernathey_profile(
        z_full, config.delta_T, config.thermocline_scale, config.H_max)

    n_cells = mesh.nCells
    nlev = z_coord.n_levels
    T_data = np.zeros((n_cells, nlev), dtype=np.float64)
    for k in range(nlev):
        T_data[:, k] = T_profile[k] * land_mask

    rng = np.random.default_rng(seed=42)
    T_data += config.T_perturbation_K * rng.standard_normal(T_data.shape) * land_mask[:, np.newaxis]

    state = state._replace(
        T=Field(jnp.array(T_data), name="T",
                dims=state.T.dims, units="degC"))

    # --- Gaussian ridge bathymetry ---
    lon_deg_cells = np.degrees(np.asarray(mesh.lonCell, dtype=np.float64))
    lat_center_rad = np.radians(0.5 * (config.lat_south + config.lat_north))
    km_per_deg = 111.32 * np.cos(lat_center_rad)
    sigma_deg = config.ridge_sigma_km / km_per_deg
    lon_c = config.ridge_lon_center_deg

    H_bathy = config.H_max - config.ridge_height * np.exp(
        -((lon_deg_cells - lon_c) ** 2) / (sigma_deg ** 2))

    H_bathy = np.where(land_mask > 0.5, np.maximum(H_bathy, 50.0), config.H_max)

    state = state._replace(
        H_bathy=Field(jnp.array(H_bathy), name="H_bathy",
                      dims=state.H_bathy.dims, units="m"))

    return state


# ---------------------------------------------------------------------------
# Forcings
# ---------------------------------------------------------------------------

def create_forcings(grid_type: str, grid, config: ACCChannelConfig = None):
    """Create OceanPhysicsConfig with channel_sine wind forcing.

    Dissipation is applied via config-level parameters (A_h, bottom_drag_r),
    not through the physics pipeline, matching the Eady experiment pattern.
    """
    if config is None:
        config = ACCChannelConfig()

    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.surface_forcing.config import (
        PrescribedForcingConfig, SurfaceForcingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig

    return OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="prescribed",
            prescribed=PrescribedForcingConfig(
                wind_profile="channel_sine",
                tau_max=config.tau0,
                lat_south_deg=config.lat_south,
                lat_north_deg=config.lat_north,
            ),
        ),
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="enhanced_diffusion"),
        shortwave_penetration=None,
    )


# ---------------------------------------------------------------------------
# Sponge
# ---------------------------------------------------------------------------

def create_sponge(grid_type: str, grid, z_coord,
                  config: ACCChannelConfig = None):
    """Create northern-boundary sponge restoring T toward Abernathey profile.

    Uses the existing sponge infrastructure (compute_sponge_gamma) but
    zeroes out the southern sponge so only the northern wall is active.
    """
    if config is None:
        config = ACCChannelConfig()

    from legoesm.ocean.sponge import SpongeForcing

    z_full = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    T_profile = _abernathey_profile(
        z_full, config.delta_T, config.thermocline_scale, config.H_max)

    if grid_type == "latlon_channel":
        from legoesm.ocean.sponge import compute_sponge_gamma_latlon

        gamma = compute_sponge_gamma_latlon(
            grid, config.lat_south, config.lat_north,
            width_deg=config.sponge_width_deg,
            timescale_days=config.sponge_timescale_days,
        )

        # Zero out southern sponge (keep only northern)
        lat_deg = np.degrees(np.asarray(grid.lat, dtype=np.float64))
        mid_lat = 0.5 * (config.lat_south + config.lat_north)
        for i, lat in enumerate(lat_deg):
            if lat < mid_lat:
                gamma[i, :] = 0.0

        n_lat, n_lon = grid.n_lat, grid.n_lon
        nlev = z_coord.n_levels
        T_ref = np.zeros((n_lat, n_lon, nlev), dtype=np.float64)
        S_ref = np.full((n_lat, n_lon, nlev), config.S_uniform, dtype=np.float64)
        for k in range(nlev):
            T_ref[..., k] = T_profile[k]

    elif grid_type == "mpas_channel":
        from legoesm.ocean.sponge import compute_sponge_gamma_mpas

        gamma = compute_sponge_gamma_mpas(
            grid, config.lat_south, config.lat_north,
            width_deg=config.sponge_width_deg,
            timescale_days=config.sponge_timescale_days,
        )

        lat_deg = np.degrees(np.asarray(grid.latCell, dtype=np.float64))
        mid_lat = 0.5 * (config.lat_south + config.lat_north)
        gamma = np.where(lat_deg < mid_lat, 0.0, gamma)

        n_cells = grid.nCells
        nlev = z_coord.n_levels
        T_ref = np.zeros((n_cells, nlev), dtype=np.float64)
        S_ref = np.full((n_cells, nlev), config.S_uniform, dtype=np.float64)
        for k in range(nlev):
            T_ref[..., k] = T_profile[k]

    else:
        raise ValueError(f"Unsupported grid type for sponge: {grid_type}")

    return SpongeForcing(
        gamma=jnp.array(gamma),
        T_ref=jnp.array(T_ref),
        S_ref=jnp.array(S_ref),
    )


# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------

EXPERIMENT_CONFIG = {
    "name": "acc_channel",
    "description": "ACC-like channel with Gaussian ridge and wind forcing",
    "scientific_purpose": "Validates flow over variable bathymetry in z-star",
    "reference": "Zhang et al. (2024), DOI: 10.1175/JPO-D-23-0042.1",
    "config_class": ACCChannelConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "default_duration": 30.0,
    "quick_duration": 2.0,
    "grid_support": {
        "latlon_channel": True,
        "mpas_channel": True,
    },
    "special_config": {
        "nlev": 20,
        "H_max": 3000.0,
        "variable_bathymetry": True,
    },
}
