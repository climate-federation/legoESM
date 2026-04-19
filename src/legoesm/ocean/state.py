"""Ocean state containers.

All states are registered as JAX pytrees via NamedTuple + Field,
consistent with ShallowWaterState, HydrostaticState, and
NonHydrostaticState in core/state.py.
"""

from __future__ import annotations

from typing import NamedTuple

from legoesm.core.field import Field


# ==============================================================================
# FV Ocean State (cubed-sphere)
# ==============================================================================

class OceanState(NamedTuple):
    """State for the ocean primitive equations on the cubed-sphere.

    3D fields: shape (6, n, n, nlev).
    2D fields: shape (6, n, n).

    Fields
    ------
    u : Field
        Zonal velocity [m/s]. Prognostic. Shape (6, n, n, nlev).
    v : Field
        Meridional velocity [m/s]. Prognostic. Shape (6, n, n, nlev).
    T : Field
        Potential temperature [degC]. Prognostic. Shape (6, n, n, nlev).
    S : Field
        Salinity [PSU]. Prognostic. Shape (6, n, n, nlev).
    eta : Field
        Sea surface height [m]. Prognostic. Shape (6, n, n).
    H_bathy : Field
        Bathymetry depth [m]. Static (positive downward). Shape (6, n, n).
    land_mask : Field
        Ocean mask. Static. 1=ocean, 0=land. Shape (6, n, n).
    """
    u: Field
    v: Field
    T: Field
    S: Field
    eta: Field
    H_bathy: Field
    land_mask: Field


class OceanTendencies(NamedTuple):
    """Tendencies for the ocean primitive equations.

    Same structure as OceanState. Static fields (H_bathy, land_mask)
    have zero tendencies, matching the phis pattern in the atmosphere.
    """
    du_dt: Field
    dv_dt: Field
    dT_dt: Field
    dS_dt: Field
    deta_dt: Field
    dH_bathy_dt: Field
    dland_mask_dt: Field


class OceanSurfaceForcing(NamedTuple):
    """External atmospheric/surface forcing data for ocean physics.

    Carries coupler-provided fields into the ocean physics pipeline.
    All fields are optional (None means not available).  Shape of 2D
    fields matches the horizontal grid; 3D fields add a level axis.

    Fields
    ------
    sw_down : array or None
        Downwelling shortwave at sea surface [W/m²].  Needed for
        subsurface SW penetration heating.
    q_net : array or None
        Net surface heat flux (positive into ocean) [W/m²].
    tau_x, tau_y : array or None
        Surface wind stress components [Pa].
    freshwater : array or None
        Net freshwater flux into ocean (P - E + R + M) [kg/m²/s].
    """
    sw_down: object = None       # jnp.ndarray | None
    q_net: object = None         # jnp.ndarray | None
    tau_x: object = None         # jnp.ndarray | None
    tau_y: object = None         # jnp.ndarray | None
    freshwater: object = None    # jnp.ndarray | None


class OceanConfig(NamedTuple):
    """Configuration for the ocean model."""
    g: float = 9.80616           # = constants.g
    rho_0: float = 1025.0        # = eos.rho_0
    A_h: float = 1.0e4           # Horizontal viscosity [m^2/s]
    K_h: float = 0.0           # Horizontal tracer diffusivity [m^2/s]
    A_v: float = 1.0e-3          # Vertical viscosity [m^2/s]
    K_v: float = 1.0e-4          # Vertical tracer diffusivity [m^2/s]
    n_barotropic_substeps: int = 30
    hyperdiff_coeff: float = 0.0
    use_conservation_fixer: bool = False
    fix_volume: bool = True
    fix_heat: bool = True
    fix_salt: bool = True
    # 2-D Laplacian damping used in barotropic subcycling.
    # Per-substep coefficient is alpha * (dt_s / dt_ref) * area * laplacian(...),
    # so the damping is explicitly dt-scaled and tunable.
    # Cubed-sphere default is higher (0.05) than other grids (0.01)
    # to suppress the face-boundary feedback instability.  See #100.
    barotropic_diffusion_alpha: float = 0.05
    barotropic_diffusion_dt_ref: float = 60.0
    # Optional runtime state checks for debugging/regression hardening.
    enable_runtime_checks: bool = False
    min_water_column_m: float = 0.5
    max_abs_eta_m: float = 1.0e4
    temperature_min_c: float = -5.0
    temperature_max_c: float = 45.0
    salinity_min_psu: float = 0.0
    salinity_max_psu: float = 50.0
    differentiable_barotropic: bool = False  # Use lax.scan (grad-compatible) vs fori_loop (faster)
    div_damp_2: float = 0.0   # 2nd-order divergence damping [m²/s] (FV discretization)
    div_damp_4: float = 0.0   # 4th-order divergence damping [m⁴/s] (FV discretization)
    physics: object = None  # OceanPhysicsConfig or None (legacy mode)
    eos: str = "wright"    # "wright" or "linear"
    eos_linear: object = None  # LinearEOSConfig when eos="linear"
    barotropic_staggering: str = "a_grid"  # "a_grid" or "c_grid" (#182)


# ==============================================================================
# Spectral Ocean State (Gaussian grid)
# ==============================================================================

class SpectralOceanState(NamedTuple):
    """State for the spectral ocean primitive equations.

    3D spectral fields: shape (n_sh, nlev) complex.
    2D spectral fields: shape (n_sh,) complex.
    Grid-space mask: shape (n_lat, n_lon) float.

    Fields
    ------
    vor_hat : Field
        Vorticity spectral coefficients. Shape (n_sh, nlev).
    div_hat : Field
        Divergence spectral coefficients. Shape (n_sh, nlev).
    T_hat : Field
        Temperature spectral coefficients. Shape (n_sh, nlev).
    S_hat : Field
        Salinity spectral coefficients. Shape (n_sh, nlev).
    eta_hat : Field
        Sea surface height spectral coefficients. Shape (n_sh,).
    H_bathy_hat : Field
        Bathymetry spectral coefficients. Static. Shape (n_sh,).
    land_mask_grid : Field
        Ocean mask in grid space. Static. Shape (n_lat, n_lon). 1=ocean.
    """
    vor_hat: Field
    div_hat: Field
    T_hat: Field
    S_hat: Field
    eta_hat: Field
    H_bathy_hat: Field
    land_mask_grid: Field


class SpectralOceanConfig(NamedTuple):
    """Configuration for the spectral ocean model."""
    g: float = 9.80616
    rho_0: float = 1025.0
    A_h: float = 1.0e4
    K_h: float = 0.0
    A_v: float = 1.0e-3
    K_v: float = 1.0e-4
    # Reserved for future split-explicit spectral stepping; currently ignored.
    n_barotropic_substeps: int = 1
    hyperdiff_coeff: float = 1.0e15
    hyperdiff_order: int = 2
    # Biharmonic hyperdiffusion on eta (SSH) to stabilise the barotropic
    # gravity-wave mode under unsplit SSP-RK3 time stepping.  Without this,
    # high-wavenumber barotropic modes exceed the RK3 imaginary-axis stability
    # limit (|omega*dt| > ~1.73) and amplify, producing SSH amplitudes ~12x
    # larger than split-explicit solvers.  Uses the same ``hyperdiff_order``
    # as the 3D fields.  Set to 0 to disable.
    eta_hyperdiff_coeff: float = 2.5e18
    use_conservation_fixer: bool = False
    min_water_column_m: float = 0.5
    time_integrator: str = "ssp_rk3"
    eos: str = "wright"
    eos_linear: object = None


# ==============================================================================
# Lat-Lon FV Ocean State
# ==============================================================================

class LatLonOceanState(NamedTuple):
    """State for the lat-lon finite-volume ocean primitive equations.

    3D fields: shape (n_lat, n_lon, nlev).
    2D fields: shape (n_lat, n_lon).

    Fields
    ------
    u : Field
        Zonal velocity [m/s]. Prognostic. Shape (n_lat, n_lon, nlev).
    v : Field
        Meridional velocity [m/s]. Prognostic. Shape (n_lat, n_lon, nlev).
    T : Field
        Potential temperature [degC]. Prognostic. Shape (n_lat, n_lon, nlev).
    S : Field
        Salinity [PSU]. Prognostic. Shape (n_lat, n_lon, nlev).
    eta : Field
        Sea surface height [m]. Prognostic. Shape (n_lat, n_lon).
    H_bathy : Field
        Bathymetry depth [m]. Static (positive downward). Shape (n_lat, n_lon).
    land_mask : Field
        Ocean mask. Static. 1=ocean, 0=land. Shape (n_lat, n_lon).
    """
    u: Field
    v: Field
    T: Field
    S: Field
    eta: Field
    H_bathy: Field
    land_mask: Field


class LatLonOceanTendencies(NamedTuple):
    """Tendencies for the lat-lon ocean primitive equations."""
    du_dt: Field
    dv_dt: Field
    dT_dt: Field
    dS_dt: Field
    deta_dt: Field
    dH_bathy_dt: Field
    dland_mask_dt: Field


class LatLonOceanConfig(NamedTuple):
    """Configuration for the lat-lon FV ocean model."""
    g: float = 9.80616
    rho_0: float = 1025.0
    A_h: float = 1.0e4           # Horizontal viscosity [m^2/s]
    K_h: float = 0.0           # Horizontal tracer diffusivity [m^2/s]
    A_v: float = 1.0e-3          # Vertical viscosity [m^2/s]
    K_v: float = 1.0e-4          # Vertical tracer diffusivity [m^2/s]
    n_barotropic_substeps: int = 30
    hyperdiff_coeff: float = 0.0
    use_conservation_fixer: bool = False
    fix_volume: bool = True
    fix_heat: bool = True
    fix_salt: bool = True
    barotropic_diffusion_alpha: float = 0.0
    barotropic_diffusion_dt_ref: float = 60.0
    enable_runtime_checks: bool = False
    min_water_column_m: float = 0.5
    max_abs_eta_m: float = 1.0e4
    temperature_min_c: float = -5.0
    temperature_max_c: float = 45.0
    salinity_min_psu: float = 0.0
    salinity_max_psu: float = 50.0
    differentiable_barotropic: bool = False
    physics: object = None
    eos: str = "wright"
    eos_linear: object = None


# ==============================================================================
# Lat-Lon C-Grid FV Ocean State
# ==============================================================================

class LatLonCGridOceanState(NamedTuple):
    """State for the lat-lon C-grid finite-volume ocean primitive equations.

    Velocities live on cell faces (Arakawa C-grid staggering):
    - u at east/west faces (lon interfaces): shape (n_lat, n_lon+1, nlev)
    - v at north/south faces (lat interfaces): shape (n_lat+1, n_lon, nlev)

    Scalars live at cell centers:
    - eta, T, S, H_bathy, land_mask: shape (n_lat, n_lon [, nlev])

    Face masks (u_mask, v_mask) are derived from the cell-center land_mask:
    a face is wet (mask=1) only if both adjacent cells are wet.

    Fields
    ------
    u : Field
        Zonal velocity at lon interfaces [m/s]. Shape (n_lat, n_lon+1, nlev).
    v : Field
        Meridional velocity at lat interfaces [m/s]. Shape (n_lat+1, n_lon, nlev).
    T : Field
        Potential temperature [degC]. Shape (n_lat, n_lon, nlev).
    S : Field
        Salinity [PSU]. Shape (n_lat, n_lon, nlev).
    eta : Field
        Sea surface height [m]. Shape (n_lat, n_lon).
    H_bathy : Field
        Bathymetry depth [m]. Static (positive downward). Shape (n_lat, n_lon).
    land_mask : Field
        Ocean mask at cell centers. Static. 1=ocean, 0=land. Shape (n_lat, n_lon).
    u_mask : Field
        Ocean mask at u-points (lon interfaces). Shape (n_lat, n_lon+1).
    v_mask : Field
        Ocean mask at v-points (lat interfaces). Shape (n_lat+1, n_lon).
    w : Field
        Vertical velocity [m/s]. Shape (n_lat, n_lon, nlev). Diagnostic field computed from flux divergence.
    """

    u: Field
    v: Field
    T: Field
    S: Field
    eta: Field
    H_bathy: Field
    land_mask: Field
    u_mask: Field
    v_mask: Field
    w: Field


class LatLonCGridOceanDiagnostics(NamedTuple):
    """Diagnostic fields for debugging ocean dynamics on lat-lon C-grid.
    
    These fields are computed during tendency calculation for analysis purposes
    but are not part of the prognostic state.
    
    Fields
    ------
    w : Field
        Vertical velocity at half levels [m/s]. Shape (n_lat, n_lon, nlev+1).
    w_half_ref : Field
        Reference vertical velocity in z* coordinates [m/s]. Shape (n_lat, n_lon, nlev+1).
    flux_div_k : Field
        Horizontal flux divergence per layer [m/s]. Shape (n_lat, n_lon, nlev).
    dT_dt_total : Field
        Total temperature tendency [degC/s]. Shape (n_lat, n_lon, nlev).
    dT_dt_hadv : Field
        Horizontal advection tendency [degC/s]. Shape (n_lat, n_lon, nlev).
    dT_dt_vadv : Field
        Vertical advection tendency [degC/s]. Shape (n_lat, n_lon, nlev).
    dT_dt_hdiff : Field
        Horizontal diffusion tendency [degC/s]. Shape (n_lat, n_lon, nlev).
    dT_dt_vdiff : Field
        Vertical diffusion tendency [degC/s]. Shape (n_lat, n_lon, nlev).
    dT_dt_physics : Field
        Physics tendency [degC/s]. Shape (n_lat, n_lon, nlev).
    wind_stress_x : Field
        Zonal wind stress applied [Pa]. Shape (n_lat, n_lon).
    wind_stress_y : Field  
        Meridional wind stress applied [Pa]. Shape (n_lat, n_lon).
    """
    w: Field
    w_half_ref: Field
    flux_div_k: Field
    dT_dt_total: Field
    dT_dt_hadv: Field
    dT_dt_vadv: Field
    dT_dt_hdiff: Field
    dT_dt_vdiff: Field
    dT_dt_physics: Field
    wind_stress_x: Field
    wind_stress_y: Field


class LatLonCGridOceanTendencies(NamedTuple):
    """Tendencies for the lat-lon C-grid ocean primitive equations."""

    du_dt: Field
    dv_dt: Field
    dT_dt: Field
    dS_dt: Field
    deta_dt: Field
    dH_bathy_dt: Field
    dland_mask_dt: Field


class LatLonCGridOceanConfig(NamedTuple):
    """Configuration for the lat-lon C-grid FV ocean model.

    Same parameter set as LatLonOceanConfig; kept separate for clarity
    since operator semantics differ (compact stencils vs centered).
    """

    g: float = 9.80616
    rho_0: float = 1025.0
    A_h: float = 1.0e4
    B_h: float = 0.0
    C_smag: float = 0.0
    bottom_drag_r: float = 0.0
    K_h: float = 0.0
    K_bih: float = 0.0
    A_v: float = 1.0e-3
    K_v: float = 1.0e-4
    n_barotropic_substeps: int = 30
    hyperdiff_coeff: float = 0.0
    use_conservation_fixer: bool = False
    fix_volume: bool = True
    fix_heat: bool = True
    fix_salt: bool = True
    barotropic_diffusion_alpha: float = 0.01
    barotropic_diffusion_dt_ref: float = 60.0
    barotropic_div_damp: float = 0.0  # Divergence damping on barotropic velocity (dimensionless)
    enable_runtime_checks: bool = False
    min_water_column_m: float = 0.5
    max_abs_eta_m: float = 1.0e4
    temperature_min_c: float = -5.0
    temperature_max_c: float = 45.0
    salinity_min_psu: float = 0.0
    salinity_max_psu: float = 50.0
    differentiable_barotropic: bool = False
    freshwater_closure: str = "virtual_salt_flux"
    S_ref: float = 35.0          # Reference salinity for virtual salt flux [PSU]
    tracer_advection: str = "tvd"  # "upwind" or "tvd" (Van Leer, #170)
    physics: object = None
    eos: str = "wright"
    eos_linear: object = None
