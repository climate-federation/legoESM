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
from typing import Dict, NamedTuple, Tuple

from legoesm import constants
from legoesm.constants import g
from legoesm.core.field import Field
from legoesm.ocean.experiments.idealized_ic import (
    coriolis_f,
    gaussian_lat_envelope,
)


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
    T_ref_C: float = 10.0          # reference temperature [degC]
    S_uniform: float = 35.0

    # Linear EOS
    alpha_T: float = 2.0e-4
    rho_0: float = constants.rho_ocean

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

    # Free-surface Laplacian: OFF. This channel runs the Voronoi C-grid, and
    # the filter is a collocated-grid device -- it exists to damp that grid's
    # checkerboard mode in sea surface height, and the value 0.05 is the
    # cubed-sphere default raised for a face-boundary feedback. This case
    # inherited it. The C-grid has no such null mode (it was introduced to
    # remove it), and the grid-scale mode that does go unstable here is the
    # rotational one, which carries near-zero surface-height gradient -- so
    # the filter is structurally blind to it.
    #
    # It was not harmless. At 0.05 on 70 km cells the implied diffusivity is
    # 3.5e6 m^2/s (ocean lateral diffusivity is 1e2-1e3), it moved 42.7x more
    # water than the actual flow, and it was the dominant destabiliser of this
    # case -- measured monotone: alpha 0.10 fails day 61.5, 0.05 day 67.4,
    # 0.025 day 82.3, 0.00 day 122.9. Turning it off also removes a mass
    # inconsistency: the filter's flux is not carried in the time-averaged
    # transport that advects thickness and tracers, so with it on, that
    # transport missed ~97% of the free-surface mass movement.
    #
    # Precedent: the NEMO-matching recipe, the DINO cards and
    # silvestri_baroclinic_jet all run this at 0, the last through an 80-day
    # turbulent transient with no dissipation backstop.
    #
    # NOT a cure. The case still fails at day 122.9 of 200; the residual is a
    # separate instability. Divergence damping below is KEPT -- it targets the
    # divergent grid mode directly and measurement shows it doing real work
    # (0.05 fails day 67, 0.025 and 0.0 both go non-finite sooner).
    barotropic_diffusion_alpha: float = 0.0
    barotropic_div_damp: float = 0.05
    tracer_advection: str = "tvd"     # "upwind", "tvd", "dst3", "dst3_multidim", "som"

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
        return -self.f0 * self.Lambda / (g * self.alpha_T)

    @property
    def f0(self) -> float:
        return coriolis_f(np.radians(self.lat_center))

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
        T_water_init_C=config.T_ref_C, T_deep=config.T_ref_C,
        S_uniform=config.S_uniform,
        land_mask_override=wall_mask,
    )


def _rest_state_mpas(mesh, z_coord, config):
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    state = rest_state_mpas_ocean(
        mesh, z_coord, H_max=config.H_max,
        T_water_init_C=config.T_ref_C, T_deep=config.T_ref_C,
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
    return gaussian_lat_envelope(lat_deg, config.lat_center, config.jet_width_deg)


def _set_uniform_stratification(state, z_coord, config, grid):
    """Set T = background stratification + depth-uniform meridional gradient.

    T(y,z) = T_ref_C + dTdz*z + dTdy * y_integrated_envelope(y)

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
    n_lat, _n_lon = T_data.shape[0], T_data.shape[1]

    lat_rad = np.asarray(grid.lat)
    lat_deg = np.degrees(lat_rad)
    lat_center_rad = np.radians(config.lat_center)
    R = constants.R_earth
    y = (lat_rad - lat_center_rad) * R

    envelope = _jet_envelope(lat_deg, config)
    T_anomaly = np.zeros(n_lat, dtype=np.float64)
    for i in range(1, n_lat):
        dy = y[i] - y[i - 1]
        T_anomaly[i] = T_anomaly[i - 1] + config.dTdy * 0.5 * (envelope[i] + envelope[i - 1]) * dy

    mask = np.asarray(state.land_mask.data)
    for k in range(nlev):
        T_zk = config.T_ref_C + config.dTdz * z_full[k]
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
    R = constants.R_earth
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
        T_zk = config.T_ref_C + config.dTdz * z_full[k]
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
    len(z_full)

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
    """Set edge-normal velocity from the purely-baroclinic U = Λ(z + H/2)·env(y).

    Matches the lat-lon IC (`_set_linear_shear_latlon`): removes the
    depth-mean from the shear so the initial state is purely baroclinic
    with zero barotropic velocity and zero SSH. This avoids the large
    geostrophic SSH signal (~2 m for U_surface=0.8 m/s) that would
    dominate the BCI perturbation and inject a barotropic adjustment
    shock at t=0 — particularly destabilising on sub-360° periodic
    channels where that adjustment can't spread across the globe.
    """
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

    # Purely baroclinic IC: eta = 0 (matches lat-lon IC).
    eta_data = np.zeros_like(state.eta.data, dtype=np.float64) * mask

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
    envelope = gaussian_lat_envelope(lat_rad, lat_center_rad, lat_width_rad)

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
    envelope = gaussian_lat_envelope(lat_cell, lat_center_rad, lat_width_rad)

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
        VerticalMixingConfig,
    )
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.bottom_drag.config import (
        BottomDragConfig,
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
# Corrected eddy-resolving recipe (Phase-G parts) — the AUDITED rebuild
# ---------------------------------------------------------------------------

class EadyUniformRecipe(NamedTuple):
    """All the pieces needed to run the Eady-uniform experiment — the canonical
    recipe shape (mirrors :class:`ocean.fidelity.veros_acc_recipe.ACCRecipe`):

    - ``model_config``: LatLonCGridOceanConfig with the corrected eddy-resolving
      dycore stack.
    - ``physics_config``: OceanPhysicsConfig (vertical mixing OFF — adiabatic BCI).
    - ``grid``: regional lat-lon C-grid channel (periodic-x, solid N/S walls).
    - ``z_coord``: z-star vertical coordinate.
    - ``wall_mask``: N/S wall mask from the channel grid (the Eady analogue of
      ACCRecipe.land_mask).
    - ``initial_state``: thermal-wind-balanced state with the seeded perturbation.

    Built by :func:`build_eady_uniform_setup`; the ``EadyUniformConfig`` is the
    recipe SPEC (pure config selecting blocks + physical params).
    """
    model_config: "object"          # LatLonCGridOceanConfig
    physics_config: "object"        # OceanPhysicsConfig
    grid: "object"                  # LatLonGrid
    z_coord: "object"               # OceanZStarCoordinate
    wall_mask: "object"             # jnp.ndarray
    initial_state: "object"         # LatLonCGridOceanState


def eady_uniform_model_config(
    config: "EadyUniformConfig" = None,
    *,
    physics=None,
    eos_config=None,
    gm_redi_cfg=None,
    recipe: str = "eady_weno5_v1",
    c_smag: float = None,
    c_leith: float = 0.0,
    c_smag_lap: float = 0.0,
    b_h: float = 0.0,
    smag_cfl_safety: float = 0.0,
    a_h: float = 0.0,
    momentum_advection: str = None,
    ke_gradient_scheme: str = None,
    tracer_advection: str = None,
    barotropic_solver: str = None,
):
    """Build the Eady-uniform ``LatLonCGridOceanConfig`` (corrected eddy-resolving
    dycore stack).

    Shared by ``build_eady_uniform_setup`` (driver path) and the test matrix
    (``EXPERIMENT_CONFIG["create_model_config"]``) so both exercise the SAME
    recipe. The matrix's ``latlon_channel`` field-scrape otherwise drops
    ``pgf_scheme``/``ke_gradient_scheme``/``C_leith``/``smag_cfl_safety`` and the
    rk3/ab2 integrators, silently testing a different (worse) dycore. The
    SCHEME identity comes from the named ``recipe`` (default
    ``"eady_weno5_v1"`` in the catalog ``legoesm.ocean.recipes``); the
    track-selection knobs (``momentum_advection``/``ke_gradient_scheme``/
    ``tracer_advection``/``barotropic_solver``), when not ``None``, OVERRIDE the
    recipe (e.g. select the NEMO-like track). The dissipation/setup params are
    layered on top. ``eos_config`` defaults to the analytic Eady linear EOS built
    from ``config``; GM/Redi is off (eddies resolved) unless a caller injects one.
    See ``build_eady_uniform_setup`` for the per-block rationale.
    """
    from legoesm.ocean.eos import LinearEOSConfig
    from legoesm.ocean.recipes import assemble_ocean_config, get_recipe
    from legoesm.ocean.state import LatLonCGridOceanConfig

    if config is None:
        config = EadyUniformConfig()
    if eos_config is None:
        eos_config = LinearEOSConfig(
            alpha_T=config.alpha_T, rho_ref=config.rho_0,
            T_ref=config.T_ref_C, S_ref=config.S_uniform)

    # The track knobs override the recipe's scheme identity when set.
    track = {k: v for k, v in (
        ("momentum_advection", momentum_advection),
        ("ke_gradient_scheme", ke_gradient_scheme),
        ("tracer_advection", tracer_advection),
        ("barotropic_solver", barotropic_solver)) if v is not None}
    bundle = get_recipe(recipe, "latlon")
    return assemble_ocean_config(
        bundle, LatLonCGridOceanConfig,
        overrides=track,
        physics=physics,
        eos_linear=eos_config,
        A_h=a_h, B_h=b_h,
        C_smag=(config.C_smag if c_smag is None else c_smag),
        C_leith=c_leith, C_smag_lap=c_smag_lap, smag_cfl_safety=smag_cfl_safety,
        A_v=config.A_v, K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        gm_redi=gm_redi_cfg,
    )


def build_eady_uniform_setup(*, n_lat: int, n_lon: int,
                             config: "EadyUniformConfig" = None, nlev: int = 20,
                             c_smag: float = None, c_leith: float = 0.0,
                             c_smag_lap: float = 0.0, b_h: float = 0.0,
                             smag_cfl_safety: float = 0.0, a_h: float = 0.0,
                             momentum_advection: str = "weno5",
                             ke_gradient_scheme: str = "centered",
                             tracer_advection: str = "weno5",
                             barotropic_solver: str = "implicit_cn"):
    """Assemble the Eady-uniform model with the CURRENT best-practice dycore
    stack for an eddy-resolving baroclinic-instability run.

    The legacy matrix path (``run_eady_uniform`` → ``_create_ocean_setup``) wires
    an April-2026 baseline: explicit-substep free-surface barotropic, forward-Euler
    outer + Euler tracers, vector-invariant momentum with the Hollingsworth-prone
    *centered* KE gradient, TVD tracers, a fixed (non-grid-scaling) biharmonic, the
    adcroft PGF, and a force-enabled KPP layer — none of the Phase-G improvements.
    Several of those are active eddy-resolving instability sources, so a failure
    there indicts the config, not the dycore.

    This builder selects the corrected canonical blocks (each ↔ the audit finding
    it fixes):

      * ``barotropic_solver="implicit_cn"``  — Crank-Nicolson Helmholtz, no 2Δt
        substep aliasing (vs explicit_substep barotropic grid-noise).
      * ``tracer_advection="weno5"`` + ``momentum_advection="weno5"`` — sharp,
        low-diffusion eddies; FLUX-form momentum sidesteps the Hollingsworth–
        Källberg instability of vector-invariant + centered KE gradient.
      * ``tracer_time_integrator="rk3"`` + ``outer_integrator="ab2"`` — accurate
        advection of the eddy field (vs forward-Euler smearing).
      * ``pgf_scheme="smc03"`` — Shchepetkin-McWilliams density-Jacobian PGF, low
        error on the tilted isopycnals that ARE the Eady problem (vs adcroft).
      * ``A_h=0, B_h=0, C_smag`` — scale-aware biharmonic Smagorinsky ONLY; drops
        the fixed ``B_h`` that does not scale with Δx across a resolution sweep.
      * vertical mixing OFF (``create_forcings`` → ``scheme="none"``); KPP is wrong
        physics for an adiabatic BCI. GM/Redi OFF (eddies are resolved).
      * linear EOS (analytic Eady), bottom drag, spherical metrics — unchanged.

    Returns an :class:`EadyUniformRecipe` (model_config, physics_config, grid,
    z_coord, wall_mask, initial_state) — the canonical recipe shape.
    """
    from legoesm.grids.latlon import create_regional_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star

    if config is None:
        config = EadyUniformConfig()

    grid, wall_mask = create_regional_latlon_grid(
        n_lat, n_lon, config.lat_south, config.lat_north,
        lon_west=config.lon_west, lon_east=config.lon_east, periodic_x=True)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=config.H_max)
    physics = create_forcings("latlon_channel", grid, config)   # vertical mix none → KPP OFF

    # Shared recipe factory (also the matrix's create_model_config). WENO5 track:
    # momentum_advection="weno5" (flux-form, no KE gradient). NEMO-like track:
    # momentum_advection="vector_invariant" + ke_gradient_scheme="hollingsworth"
    # (NEMO nn_dynkeg=1).
    model_config = eady_uniform_model_config(
        config, physics=physics,
        barotropic_solver=barotropic_solver,
        tracer_advection=tracer_advection,
        momentum_advection=momentum_advection,
        ke_gradient_scheme=ke_gradient_scheme,
        # --- scale-aware dissipation (tunable for eddy-resolving) ---
        # C_smag = biharmonic Smagorinsky (scale-selective); C_leith = Leith
        # (enstrophy-cascade-aware); both grid-aware so they scale across a
        # resolution sweep. c_smag_lap/b_h available for extra grid-scale control.
        # VALIDATED EDDY-RESOLVING MINIMUM-DISSIPATION RECIPE (≥120×120, weak U=0.2,
        # dt=600; ralph-loop search, docs/dev-notes/planning/eady_eddy_resolving_ralph.md):
        # the COMBINATION a_h≈1000 + c_smag≈0.1 + smag_cfl_safety=0.5 is stable,
        # spectrally clean, and keeps strong eddies (pure A_h over-damps; pure
        # biharmonic Smagorinsky blows up at 120 — the ~4–5Δx mode is too close to
        # the eddy scale). (pgf_scheme="smc03", rk3/ab2, linear EOS, GM/Redi off all
        # fixed inside eady_uniform_model_config.)
        c_smag=c_smag, c_leith=c_leith, c_smag_lap=c_smag_lap,
        b_h=b_h, smag_cfl_safety=smag_cfl_safety, a_h=a_h,
    )
    initial_state = create_initial_conditions("latlon_channel", grid, z_coord, config)
    return EadyUniformRecipe(
        model_config=model_config, physics_config=physics, grid=grid,
        z_coord=z_coord, wall_mask=wall_mask, initial_state=initial_state)


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
    "create_model_config": eady_uniform_model_config,
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
