"""Ocean state containers.

All states are registered as JAX pytrees via NamedTuple + Field,
consistent with ShallowWaterState, HydrostaticState, and
NonHydrostaticState in core/state.py.
"""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants
from legoesm.core.field import Field
from legoesm.ocean.biogeochemistry.config import OceanBiogeoState


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

    K_v / A_v are optional interface-level diffusivity / viscosity
    profiles populated by the physics function when
    ``implicit_vertical_mixing`` is enabled.  Shape
    ``(..., nlev-1)`` at interior interfaces, or None.
    """
    du_dt: Field
    dv_dt: Field
    dT_dt: Field
    dS_dt: Field
    deta_dt: Field
    dH_bathy_dt: Field
    dland_mask_dt: Field
    K_v: object = None   # tracer diffusivity at interfaces [m²/s]
    A_v: object = None   # momentum viscosity at interfaces [m²/s]


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
    g: float = constants.g
    rho_0: float = constants.rho_ocean
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
    # Bottom drag (mirror of LatLonCGridOceanConfig). ``bottom_drag_r > 0``
    # enables linear drag ``du/dt|_drag = -r*u/h_bot`` on the bottom
    # cell; setting ``bottom_drag_bg_velocity > 0`` lifts it to the
    # MOM6 quadratic-with-floor form. ``bottom_drag_bbl_thickness > 0``
    # spreads the drag over a fixed Ekman thickness (Killworth &
    # Edwards 1999) instead of dumping it into a possibly very thin
    # partial cell.
    bottom_drag_r: float = 0.0
    bottom_drag_bg_velocity: float = 0.0
    bottom_drag_bbl_thickness: float = 0.0
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
    g: float = constants.g
    rho_0: float = constants.rho_ocean
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
    g: float = constants.g
    rho_0: float = constants.rho_ocean
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
    T_som : Field or None
        SOM (Prather 1986) moments for temperature. Shape (n_lat, n_lon, nlev, 9).
        Order: [sx, sy, sz, sxx, syy, szz, sxy, sxz, syz].
        None when tracer_advection != "som".
    S_som : Field or None
        SOM (Prather 1986) moments for salinity. Same shape and order as T_som.
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
    T_som: object = None
    S_som: object = None
    biogeo: "OceanBiogeoState | None" = None
    T_flux_div_prev: object = None  # Previous advection flux divergence for T (AB2 only)
    S_flux_div_prev: object = None  # Previous advection flux divergence for S (AB2 only)


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
    """Tendencies for the lat-lon C-grid ocean primitive equations.

    K_v / A_v are optional interface-level diffusivity / viscosity
    profiles populated when ``implicit_vertical_mixing`` is enabled.
    """
    du_dt: Field
    dv_dt: Field
    dT_dt: Field
    dS_dt: Field
    deta_dt: Field
    dH_bathy_dt: Field
    dland_mask_dt: Field
    K_v: object = None
    A_v: object = None


class MomentumTendencyDiagnostics(NamedTuple):
    """Per-term momentum-tendency diagnostics for closure analysis.

    Captured at the point each term is computed inside
    ``compute_tendencies`` so that, by construction,
    ``Σ components == du_dt`` to machine precision.  Mirrors the
    pattern in MOM6 (``MOM_diagnostics``), MITgcm
    (``DIAGNOSTICS_PKG``), and NEMO (``trd_*``).

    All fields share the same shape as ``du_dt`` / ``dv_dt`` (3D on the
    C-grid u-/v-faces).  Terms not active in a given config (e.g.,
    biharmonic when ``B_h == 0``) are zero arrays.

    Naming convention: ``<term>_u`` and ``<term>_v`` for the u- and
    v-momentum contributions respectively.  The same pattern can be
    re-used for future tracer or energy budgets — see Phase 1.5 of
    docs/ocean_experiments/global_overturning_plan.md.

    Fields
    ------
    KE_PGF_u, KE_PGF_v : Field
        −∂(KE)/∂x − (1/ρ_0)·∂p/∂x   (kinetic-energy gradient + pressure gradient)
    vortcor_u, vortcor_v : Field
        ζ × v_at_u  /  −ζ × u_at_v   (RELATIVE vorticity advection only;
        the planetary Coriolis f×u is applied in the forward-backward step
        function and is NOT included in these diagnostics)
    vertadv_u, vertadv_v : Field
        Flux-form 1st-order upwind ∂(w·u)/∂z, ∂(w·v)/∂z
    Ah_lap_u, Ah_lap_v : Field
        A_h · ∇²u_prime, A_h · ∇²v_prime  (lateral Laplacian viscosity
        on the *baroclinic perturbation*; depth integral is identically 0)
    Bh_bilap_u, Bh_bilap_v : Field
        −B_h · scale · ∇⁴u_prime  (biharmonic, 0 when B_h = 0)
    Cs_smag_u, Cs_smag_v : Field
        Smagorinsky biharmonic (0 when C_smag = 0)
    Cl_leith_u, Cl_leith_v : Field
        Leith biharmonic (0 when C_leith = 0)
    botdrag_u, botdrag_v : Field
        −r·u/dz_bot at the bottom level only; zero elsewhere
        (matches the model's path-1 explicit bottom-cell drag)
    Av_vert_u, Av_vert_v : Field
        A_v · ∂²u/∂z² (vertical viscosity on the perturbation)
    phys_u, phys_v : Field
        ``phys.du_dt`` / ``phys.dv_dt`` from the surface-forcing physics
        module (wind stress at the surface; possibly other physics
        contributions if active)
    sponge_u, sponge_v : Field
        Sponge restoring (0 when no sponge)
    total_u, total_v : Field
        The du_dt / dv_dt returned by the PE tendency function (after
        mask multiplication).  **Excludes Coriolis** (f×u), which is
        applied in the forward-backward step function.  Sanity check:
        ``total ≡ Σ PE components`` to machine precision — enforced by
        ``test_momentum_diagnostics_closure``.
    """

    KE_PGF_u: Field
    KE_PGF_v: Field
    vortcor_u: Field
    vortcor_v: Field
    Dterm_u: Field        # WENO momentum-advection D-term (Silvestri 2024
    Dterm_v: Field        # Eqs. 31-32); zero unless `momentum_advection`
                          # in {"weno5","weno7"} and `weno_d_term=True`.
    vertadv_u: Field
    vertadv_v: Field
    Ah_lap_u: Field
    Ah_lap_v: Field
    Bh_bilap_u: Field
    Bh_bilap_v: Field
    Cs_smag_u: Field
    Cs_smag_v: Field
    Cl_leith_u: Field
    Cl_leith_v: Field
    botdrag_u: Field
    botdrag_v: Field
    Av_vert_u: Field
    Av_vert_v: Field
    phys_u: Field
    phys_v: Field
    sponge_u: Field
    sponge_v: Field
    total_u: Field
    total_v: Field


class LatLonCGridOceanConfig(NamedTuple):
    """Configuration for the lat-lon C-grid FV ocean model.

    Same parameter set as LatLonOceanConfig; kept separate for clarity
    since operator semantics differ (compact stencils vs centered).
    """

    g: float = constants.g
    rho_0: float = constants.rho_ocean
    A_h: float = 1.0e4
    A_h_lat_scaling: bool = False  # When True, A_h is scaled by cos(lat) to
                                    # keep the grid Reynolds number latitude-
                                    # independent on lat-lon grids.  Default
                                    # False to preserve bit-exact regression on
                                    # legacy configs.
    A_h_floor: float = 0.0         # Minimum effective A_h [m²/s] after latitude
                                    # scaling.  Prevents viscosity from vanishing
                                    # at extreme latitudes.  Recommended 1000.0
                                    # for grids extending past 85°.
    A_h_eq_boost: float = 1.0      # Equatorial Laplacian-viscosity boost.  When
                                    # > 1, multiplies A_h by 1 + (boost-1) *
                                    # exp(-(lat/sigma)²), so horizontal momentum
                                    # gets extra dissipation near the equator
                                    # where f→0 leaves no rotational stiffness.
                                    # Targets unconstrained equatorial dynamic
                                    # response at coarse resolution that drives
                                    # runaway upwelling cold tongues.  Typical
                                    # production: 3-10.  Only multiplies A_h
                                    # (momentum); K_h (tracers) is untouched
                                    # so water masses stay intact.
    A_h_eq_sigma_deg: float = 5.0  # Gaussian half-width in degrees of the
                                    # equatorial boost.  Typical 3-7°
                                    # (~equatorial waveguide width).
    A_h_merid: float = 0.0        # Meridional-only Laplacian viscosity [m²/s].
                                    # Applies d²u/dy² directly at u-faces and
                                    # d²v/dy² at v-faces — a scalar operator
                                    # that damps meridional structure (2Δy mode)
                                    # without affecting zonal flow.  Independent
                                    # of A_h.  Use on lat-lon grids where
                                    # dx/dy anisotropy makes isotropic A_h
                                    # either too strong (zonal) or too weak
                                    # (meridional).
    B_h: float = 0.0
    B_h_lat_scaling: bool = True   # Apply (cos(lat)/cos_max)⁴ scaling to B_h.
                                    # Default True (MOM6 convention) prevents
                                    # CFL violation at poles where dx shrinks.
                                    # Set False to keep full B_h everywhere
                                    # (requires smaller dt for CFL safety).
    B_h_barotropic: float = 0.0  # Biharmonic hyperviscosity coeff [m^4/s]
                                   # applied to the DEPTH-MEAN (U_bar,
                                   # V_bar) only, via the F_slow channel
                                   # of the implicit-CN barotropic
                                   # solver.  Damps the barotropic
                                   # standing mode at deep cells next
                                   # to steep slopes without touching
                                   # baroclinic geostrophy (which lives
                                   # in u' = u_3d - U_bar).  HIM/MOM6
                                   # BIHARMONIC_BAROTROPIC analog.
    C_smag: float = 0.0            # Biharmonic Smagorinsky coefficient
    C_smag_lap: float = 0.0        # Laplacian Smagorinsky coefficient.
                                    # When > 0, adds flow-adaptive Laplacian
                                    # viscosity A_smag = (C·dx)²·|D| via the
                                    # energy-stable stress-tensor operator.
                                    # MOM6 OM4 uses 0.15. Additive with A_h.
    bottom_drag_r: float = 0.0
    bottom_drag_bbl_thickness: float = 0.0
    bottom_drag_bg_velocity: float = 0.0  # MOM6 DRAG_BG_VEL [m/s]; when >0,
                                           # drag is quadratic-with-floor:
                                           # tau ∝ √(u²+v²+u_bg²) · u, with
                                           # the linear-in-u limit set to
                                           # bottom_drag_r at |u|→0.
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
    bebt: float = 0.2               # Semi-implicit barotropic PGF [0,1]. 0=forward-backward, 0.2=MOM6 default.
    maxvel_barotropic: float = 0.0  # Velocity clipping [m/s]. 0=disabled. MOM6 uses 6.0.
    barotropic_time_filter: str = "cosine"  # "box" or "cosine" (shaped filter for time-averaging)
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
    tracer_advection: str = "tvd"  # "upwind", "tvd", "ppm_fct", "ppm", "dst3", "dst3_multidim", "som", "weno5", "weno7"
    gm_redi: object = None         # GMRediConfig or None; enables GM/Redi lateral mixing
    physics: object = None
    eos: str = "wright"
    eos_linear: object = None
    # Leith viscosity coefficient (Leith 1996).  When > 0 enables
    # flow-adaptive biharmonic viscosity ``-∇²(A_L ∇²u)`` with
    # ``A_L = (C_L · Δ)³ · |∇ζ|`` (or ``sqrt(|∇ζ|² + |∇δ|²)`` when
    # ``C_leith_modified = True``).  Typical values: 1.0–2.0.  Appended
    # at the END of the NamedTuple so existing positional call sites
    # keep working.
    C_leith: float = 0.0
    C_leith_modified: bool = False
    # Slope-foot viscosity enhancement (MOM6 OM4 KH_BG_2D analog).
    # When > 0, multiplies horizontal viscosity (A_h Laplacian, Smagorinsky,
    # Leith) in the bottom N levels by 1 + alpha · tanh(|∇H|/H/δ),
    # locally enhancing dissipation over steep slopes (African shelf,
    # ITF, equatorial trenches). Targets the f≈0 + steep-bathymetry
    # instability mode that constant viscosity cannot reach.
    slope_foot_alpha: float = 0.0       # 0 = disabled; production: 3.0
    slope_foot_threshold: float = 0.1   # MOM6 default
    slope_foot_n_levels: int = 5        # bottom 5 levels
    momentum_advection: str = "vector_invariant"  # "vector_invariant", "weno5", or "weno7"
    # Kinetic-energy gradient scheme for the vector-invariant form.
    # ``"centered"`` (default; legacy bit-exact): legoESM's existing
    # ``KE = 0.5·((⟨u⟩ᵢ)² + (⟨v⟩ⱼ)²)`` form. The standard centered
    # C-grid scheme suffers the Hollingsworth-Kallberg instability over
    # stratified flow on sloping bathymetry (see #263).
    # ``"hollingsworth"``: NEMO 4.2.1 dynkeg.F90 ``nkeg_HW`` form
    # (Hollingsworth, Kållberg & Renner 1983; Arakawa & Hsu 1990).
    # Wider (3-row) stencil that smooths spurious vortex stretching.
    # Strongly recommended for stratified ocean over realistic
    # bathymetry (NEMO turns this on by default via ``nn_dynkeg=1``).
    ke_gradient_scheme: str = "centered"
    weno_d_term: bool = True  # Include WENO D-term (divergence flux, Silvestri Eqs. 31-32).
                              # Implemented with proper split: matching-direction divergence
                              # is WENO-upwinded, cross-direction stays centered (Appendix C).
                              # Set False to disable the divergent-mode dissipation.
    # Barotropic solver selection (see docs/issues/barotropic_mode_noise.md).
    # ``"explicit_substep"`` (default) → existing forward-backward substep
    # loop with cosine/box time filter.
    # ``"implicit_cn"`` → single-step Crank-Nicolson free surface, PCG
    # Helmholtz solve.  Eliminates the chequerboard mode by construction;
    # no substepping, no time filter, no divergence damping needed.
    barotropic_solver: str = "explicit_substep"
    # Implicit-solver knobs (only used when ``barotropic_solver = 'implicit_cn'``).
    # 0.5 = pure Crank-Nicolson (2nd-order, no implicit damping); 1.0 =
    # fully backward (1st-order, maximum damping).  0.55 is the standard
    # MITgcm/MPAS-O choice — slightly past CN for chequerboard suppression
    # while staying close to 2nd-order in time.
    barotropic_implicit_theta_eta: float = 0.55
    barotropic_implicit_theta_pgf: float = 0.55
    barotropic_implicit_pcg_tol: float = 1.0e-10
    barotropic_implicit_pcg_maxiter: int = 200
    # Pressure-gradient force scheme on partial cells.  ``"adcroft"``
    # (default): existing centered-diff p_prime + Adcroft & Campin 2004
    # face-PGF correction.  ``"smc03"``: full Shchepetkin & McWilliams
    # 2003 density-Jacobian PGF with harmonic-mean monotonized slope
    # reconstruction — closes the BH partial-cell gap by avoiding the
    # single-level z-spike that the Adcroft correction produces and that
    # drives the 2Δz computational mode.  See
    # docs/ocean_experiments/density_jacobian_pgf_plan.md.  Pure-z*
    # runs ignore this field (the existing path is identical).
    pgf_scheme: str = "adcroft"
    # Tracer time integration for the flux-form advection step.
    # "euler" (default): forward Euler (1st-order).
    # "ab2": Adams-Bashforth 2 with stabilization (MITgcm convention).
    #   Formally 1st-order when ab2_epsilon > 0, but error constant is
    #   ~epsilon * dt, much smaller than Euler's ~dt.  Set ab2_epsilon=0
    #   for pure 2nd-order (less stable).  CFL limit is ~0.72 (tighter
    #   than Euler's ~1.0).
    # "rk3": RK3 in Butcher-tableau form (3rd-order, 3x advection cost).
    #   Uses effective tendency F_eff = F0/6 + F1/6 + 2*F2/3 for exact
    #   conservation.  Note: the Shu-Osher SSP (monotonicity) property
    #   is NOT preserved in this form — new extrema may appear with
    #   nonlinear limiters (TVD, WENO, FCT).
    tracer_time_integrator: str = "euler"
    ab2_epsilon: float = 0.1  # AB2 stabilization (MITgcm ABepsBar)
    # Implicit (backward-Euler) vertical mixing.  When True (default):
    #   1. The PE tendency function skips the explicit ``A_v`` viscous
    #      block (lines tagged ``if config.A_v > 0 ...``).
    #   2. The vertical-mixing and ``enhanced_diffusion`` convection
    #      schemes are called with ``apply_diffusion=False`` — they
    #      return zero local-diffusion tendency but still produce the
    #      K_v / A_v profile and (for KPP) the non-local counter-
    #      gradient flux.
    #   3. After the barotropic step, tracer advection, and GM/Redi,
    #      the model step applies an unconditionally-stable backward-
    #      Euler tridiagonal solve to ``T, S, u, v`` using the summed
    #      K_v / A_v profiles.
    # Removes the explicit-diffusion CFL limit ``dt < dz² / (2 K)``,
    # which becomes binding when ``K_conv = 1 m²/s`` is active with
    # surface dz < 30 m or when vertical resolution is increased.
    # MOM6 / NEMO / POP / MITgcm all use this approach, and MPAS
    # (`MPASOceanConfig`) also defaults True. The lat-lon default was
    # flipped from False to True on 2026-05-14 after a DINO forced run
    # at j=0,i=26 hit the explicit-CFL bound under KPP-driven cold
    # restoring (see docs/ocean_experiments/dino_replication_plan.md
    # Finding 5). Set explicit ``implicit_vertical_mixing=False`` to
    # reproduce the historical explicit-diffusion behavior.
    implicit_vertical_mixing: bool = True
