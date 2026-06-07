"""Ocean state containers.

All states are registered as JAX pytrees via NamedTuple + Field,
consistent with ShallowWaterState, HydrostaticState, and
NonHydrostaticState in core/state.py.
"""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants
from legoesm.core.field import Field
from legoesm.ocean.constants_config import ConstantsConfig

# Seawater freezing point in degC (model T is in degC), captured at MODULE scope
# where ``constants`` is the module.  Inside ``LatLonCGridOceanConfig`` the field
# ``constants: ConstantsConfig`` (defined mid-class) shadows the module name, so a
# field default cannot evaluate ``constants.T_freeze_ocean`` directly -- reference
# this module-level value instead.  = 271.35 - 273.15 = -1.8 C (no literal).
_T_FREEZE_OCEAN_C: float = constants.T_freeze_ocean - constants.T_freeze


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
    salt_flux : array or None
        REAL salt-mass flux into the ocean [kg(salt)/m²/s, positive = salt INTO
        ocean], e.g. sea-ice brine rejection on freeze.  Applied to the top
        layer salinity as dS/dt = salt_flux*1e3/(rho_0*dz_0); distinct from the
        ``freshwater`` (virtual-salt dilution) channel.
    """
    sw_down: object = None       # jnp.ndarray | None
    q_net: object = None         # jnp.ndarray | None
    tau_x: object = None         # jnp.ndarray | None
    tau_y: object = None         # jnp.ndarray | None
    freshwater: object = None    # jnp.ndarray | None
    salt_flux: object = None     # jnp.ndarray | None  (real salt mass, kg/m2/s)


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
    # Baroclinic time integrator for the explicit (slow) momentum+tracer update
    # in OceanModel.step.  Forward-Euler (default) cannot carry a realistic WOA
    # cold-start (violent geostrophic adjustment -> grid-scale blowup, the same
    # gap RK3 closed for the lat-lon C-grid tripole). "rk3" wraps the baroclinic
    # update in 3-stage SSP-RK3 (3x tendency cost); the barotropic substeps run
    # once afterward as before.
    baroclinic_rk3: bool = False
    # Per-step velocity ceiling [m/s] (0 = off). Pragmatic stabiliser for the
    # quasi-uniform cube: sub-grid marginal-sea fronts spin up uncarriable jets;
    # clipping |u|,|v| bounds them (open ocean untouched) for a caveated comparison.
    velocity_ceiling: float = 0.0
    eos: str = "wright"    # "wright" or "linear"
    eos_linear: object = None  # LinearEOSConfig when eos="linear"
    # Partial-cell horizontal pressure-gradient scheme on the cd-grid AL
    # corners (only active with an OceanPartialCellCoordinate; pure-z* runs
    # ignore it and stay bit-exact).  ``"adcroft"``: Adcroft & Campin 2004
    # LINEAR depth-shift correction added to the plain AL gradient of p'.
    # ``"smc03"``: full Shchepetkin & McWilliams 2003 density-Jacobian PGF
    # (harmonic-mean monotonized slope reconstruction).  The linear Adcroft
    # correction leaves a 2nd-order residual for a vertically-stratified
    # rest column over partial topography (the spurious bottom-trapped PGF
    # that seeds the cube cold-start blowup); smc03 reconstructs ρ(z) so
    # adjacent shifted-centroid columns evaluate pressure identically and
    # the rest-state PGF vanishes.  Mirrors the proven latlon/tripole path
    # (which default to "smc03").
    pgf_scheme: str = "adcroft"
    barotropic_staggering: str = "a_grid"  # "a_grid", "c_grid" or "fv3sw"
    # Damping for the "fv3sw" barotropic (the validated cube SW core).  The
    # atmosphere-calibrated iter1009 preset (div_damp_factor=8) is too weak for
    # the slower ocean barotropic, so phillips_two_layer over-grows (eta_growth
    # 24 vs latlon/mpas ~4.5).  div_damp_factor=120 is ocean-tuned at C24 so
    # phillips matches latlon/mpas (max_eta 0.44 vs latlon 0.41) WHILE keeping
    # barotropic_wave (0.86 m) and geostrophic_adjustment (max_speed 0.014 ≈
    # latlon) un-over-damped.  Only used when barotropic_staggering=="fv3sw".
    barotropic_sw_div_damp_factor: float = 120.0
    barotropic_sw_damp_v: float = 0.030


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
    T_flux_div_prev: object = None  # Previous advection flux divergence for T (AB2 only)
    S_flux_div_prev: object = None  # Previous advection flux divergence for S (AB2 only)
    # Prognostic eddy kinetic energy [m^2/s^2], 2-D Field (n_lat, n_lon), used only
    # when the prognostic-EKE GM closure is active (config.gm_redi.eke not None).
    # Default None -> inert (no EKE): zero behaviour change for existing configs.
    eke: object = None
    # Prior EXPLICIT increment ΔX_expl^{n-1} for the AB2 outer integrator
    # (config.outer_integrator == "ab2"): the explicit-only forward-Euler increment
    # (advection, GM/Redi, lateral friction, Coriolis, barotropic solve, freshwater)
    # WITHOUT the implicit vertical mixing — implicit mixing is applied once, after
    # the AB2 extrapolation, and is NOT carried. Tracers store the full explicit
    # increment; u/v store the BAROCLINIC-deviation explicit increment (the barotropic
    # mode is kept from the barotropic solve, un-AB2'd). Default None -> inert
    # (forward-Euler): zero behaviour change.
    T_incr_prev: object = None
    S_incr_prev: object = None
    u_incr_prev: object = None
    v_incr_prev: object = None
    # Rigid-lid barotropic streamfunction state (config.barotropic_solver ==
    # "rigid_lid").  ``psi`` is the vertex-point streamfunction [m^3/s], shape
    # (n_lat+1, n_lon+1).  ``dpsi``/``dpsi_prev`` are the interior streamfunction
    # tendencies ∂ψ/∂t at the current/previous solved step (the AB2 history +
    # the leapfrog CG-guess history).  ``dpsin``/``dpsin_prev`` are the
    # per-island streamfunction-constant tendencies, shape (nisle,).  All default
    # None -> inert (free-surface path unaffected; zero behaviour change).
    psi: object = None
    dpsi: object = None
    dpsi_prev: object = None
    dpsin: object = None
    dpsin_prev: object = None


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


class SurfaceTracerForcing(NamedTuple):
    """Surface TRACER forcing RATE (restoring + prescribed q_net + penetrating
    shortwave) WITHHELD from the explicit ``dT_dt``/``dS_dt`` so the model step
    can apply it IMPLICITLY (weight 1.0, no AB2 extrapolation) inside the
    backward-Euler vertical-mixing solve — matching Veros's
    ``forc_temp_surface``/``forc_salt_surface`` placement
    (``veros/core/thermodynamics.py``: the surface forcing enters the implicit
    vertical-diffusion tridiagonal RHS, not the explicit AB2 tendency).

    Populated ONLY when ``LatLonCGridOceanConfig.surface_forcing_implicit`` is
    True; ``None`` otherwise (default), keeping the tendency pytree + every
    existing path bit-identical.

    Fields
    ------
    dT_dt : Field
        Surface temperature forcing rate [degC/s], full-column shape
        ``(n_lat, n_lon, nlev)`` — nonzero in the surface layer (index 0) for
        the restoring + non-solar q_net, plus the shortwave-penetration column.
    dS_dt : Field
        Surface salinity forcing rate [PSU/s], full-column shape; surface-layer
        restoring only (Veros ACC does not restore salinity).
    """
    dT_dt: Field
    dS_dt: Field


class LatLonCGridOceanTendencies(NamedTuple):
    """Tendencies for the lat-lon C-grid ocean primitive equations.

    K_v / A_v are optional interface-level diffusivity / viscosity
    profiles populated when ``implicit_vertical_mixing`` is enabled.

    Ah_visc_u / Ah_visc_v are the harmonic LATERAL-viscosity momentum
    tendencies (``A_h∇²u`` / ``A_h∇²v`` [m/s²], face-masked), populated
    ONLY when the prognostic-EKE ``source_kdiss_h`` option is on (so the
    3-D EKE step can route the mean-KE they remove into the EKE source —
    Veros's ``K_diss_h``, the DYNAMICAL-form path). ``None`` otherwise
    (default), keeping the tendency pytree + every existing path bit-identical.

    Ah_kediss_cell is the FAITHFUL positive-definite K_diss_h dissipation
    density (``A_h·(div² + <ζ²>)`` [m²/s³] at cell centres), populated ONLY
    when ``source_kdiss_h`` AND ``kdiss_h_flux_form`` are both on (the ACC
    recipe). It is the Helmholtz KE-removal of the vector-Laplacian lateral
    viscosity — ≥ 0 everywhere by construction (no clamp). ``None`` otherwise.

    surface_tracer_forcing is a :class:`SurfaceTracerForcing` (dT/dS rate)
    populated ONLY when ``surface_forcing_implicit`` is on (the ACC recipe): the
    surface TRACER forcing (restoring + q_net + shortwave penetration) is then
    WITHHELD from ``dT_dt``/``dS_dt`` and applied at weight 1.0 inside the
    backward-Euler implicit vertical-mixing solve (Veros placement). ``None``
    otherwise (default), keeping every existing path bit-identical.
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
    Ah_visc_u: object = None
    Ah_visc_v: object = None
    Ah_kediss_cell: object = None
    surface_tracer_forcing: object = None


class MomentumTendencyDiagnostics(NamedTuple):
    """Per-term momentum-tendency diagnostics for closure analysis.

    Captured at the point each term is computed inside
    ``compute_tendencies`` so that, by construction,
    ``Σ components == du_dt`` to machine precision.  Mirrors the
    pattern in MOM6 (``MOM_diagnostics``), MITgcm
    (``DIAGNOSTICS_PKG``), and NEMO (``trd_*``).

    Exception — ``adaptive_implicit_vertadv=True``: vertical advection is
    applied as a separate operator-split stage at the step level (NEMO
    ``ln_zad_Aimp``), so ``vertadv_{u,v}`` is then a DIAGNOSTIC-ONLY
    start-of-step estimate that is NOT in ``du_dt``; the closure becomes
    ``Σ (components except vertadv) == du_dt``.

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
        Flux-form 1st-order upwind ∂(w·u)/∂z, ∂(w·v)/∂z.  With
        ``adaptive_implicit_vertadv=True`` this holds the start-of-step
        explicit estimate only (the actual term is applied implicitly at
        the step level and is excluded from ``du_dt``); see the class
        docstring closure note.
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

    ~70 fields. This docstring is the *map* the flat field list lacks; the
    fields group as:

    - **Physical constants**: ``g``, ``rho_0``, ``constants`` (ConstantsConfig).
    - **Lateral (harmonic) viscosity**: ``A_h`` + ``A_h_lat_scaling``,
      ``A_h_cos_power``, ``A_h_floor``, ``A_h_eq_boost``/``A_h_eq_sigma_deg``,
      ``A_h_merid``, ``A_h_cap_*``.
    - **Biharmonic viscosity**: ``B_h`` + ``B_h_lat_scaling``, ``B_h_barotropic``.
    - **Eddy-viscosity closures**: ``C_smag``, ``C_smag_lap``, ``C_leith``,
      ``C_leith_modified``, ``slope_foot_*``.
    - **Tracer diffusivity / vertical mixing**: ``K_h``, ``K_bih``, ``A_v``,
      ``K_v``, ``implicit_vertical_mixing``.
    - **Bottom drag**: ``bottom_drag_r`` + ``bottom_drag_bbl_thickness``,
      ``bottom_drag_bg_velocity`` (physics-level BottomDragConfig is deprecated).
    - **Barotropic solver**: ``barotropic_solver``, ``n_barotropic_substeps``,
      ``bebt``, ``barotropic_div_damp``, ``barotropic_diffusion_*``,
      ``maxvel_barotropic``, ``barotropic_time_filter``, ``barotropic_implicit_*``,
      ``differentiable_barotropic``.
    - **Numerics choices**: ``tracer_advection``, ``momentum_advection``,
      ``ke_gradient_scheme``, ``weno_d_term``, ``pgf_scheme``,
      ``tracer_time_integrator``/``ab2_epsilon``, ``hyperdiff_coeff``.
    - **EOS / eddy param / physics**: ``eos`` (+ ``eos_linear``), ``gm_redi``,
      ``physics`` (full OceanPhysicsConfig pipeline).
    - **Conservation & freshwater**: ``use_conservation_fixer``,
      ``fix_volume``/``fix_heat``/``fix_salt``, ``fix_eta_drift``,
      ``freshwater_closure``, ``S_ref``.
    - **Runtime invariant checks**: ``enable_runtime_checks`` + bounds
      (``min_water_column_m``, ``max_abs_eta_m``, ``temperature_min/max_c``,
      ``salinity_min/max_psu``).

    Minimal run (everything else defaults to sane Earth values)::

        cfg = LatLonCGridOceanConfig()       # constant A_v/K_v, no physics pipeline

    Production-style::

        cfg = LatLonCGridOceanConfig(
            A_h=3e4, A_h_lat_scaling=True, B_h=1e10, C_smag=0.15,
            bottom_drag_r=2.5e-3, implicit_vertical_mixing=True,
            barotropic_solver="implicit", eos="wright",
            physics=OceanPhysicsConfig(...),  # KPP/TKE + GM/Redi
        )

    Same parameter set as LatLonOceanConfig; kept separate since operator
    semantics differ (compact stencils vs centered).

    Section headers below mark the contiguous top run of fields. The trailing
    fields (from ``n_barotropic_substeps`` on) are kept in *chronological*
    append order to preserve positional construction for legacy callers, so
    they span several topics — use the group-map above to locate them, not the
    physical field order.
    """

    # --- Physical constants (defaults reference legoesm.constants; pin via
    #     ConstantsConfig for a reference-model recipe) ---
    g: float = constants.g
    rho_0: float = constants.rho_ocean

    # --- Lateral (harmonic Laplacian) viscosity ---
    A_h: float = 1.0e4
    A_h_lat_scaling: bool = False  # When True, A_h is scaled by cos(lat)^N to
                                    # keep the grid Reynolds number latitude-
                                    # independent on lat-lon grids.  Default
                                    # False to preserve bit-exact regression on
                                    # legacy configs.
    A_h_cos_power: int = 1         # Exponent N on cos(lat) used when
                                    # ``A_h_lat_scaling`` is True.  Equivalent
                                    # to Veros's ``hor_friction_cosPower``.
                                    # N=1 (constant grid Reynolds, default) is
                                    # the production choice; N=2 (constant
                                    # viscous CFL, legacy) preserves the
                                    # pre-2024 convention.
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

    # --- Biharmonic viscosity ---
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
    # --- Smagorinsky eddy viscosity ---
    C_smag: float = 0.0            # Biharmonic Smagorinsky coefficient
    C_smag_lap: float = 0.0        # Laplacian Smagorinsky coefficient.
                                    # When > 0, adds flow-adaptive Laplacian
                                    # viscosity A_smag = (C·dx)²·|D| via the
                                    # energy-stable stress-tensor operator.
                                    # MOM6 OM4 uses 0.15. Additive with A_h.
    smag_cfl_safety: float = 0.0   # When > 0, cap the Laplacian-Smagorinsky
                                    # coefficient at the per-cell tuned ceiling
                                    # ``smag_cfl_safety * area * cos^2(lat) / dt``
                                    # (see laplacian_smag_cfl_cap; the larger
                                    # stricter-than-CFL ceiling the WBC cold-
                                    # start needs) so it can be
                                    # cranked high to damp western-boundary-
                                    # current jets WITHOUT self-CFL-violating at
                                    # the sharp jet (the WBC cold-start blowup).
                                    # ~0.125 (1/8) is a safe 2-D Laplacian cap.

    # --- Bottom drag (dynamics-level; the physics-pathway BottomDragConfig is
    #     deprecated — set drag here) ---
    bottom_drag_r: float = 0.0
    bottom_drag_bbl_thickness: float = 0.0
    bottom_drag_bg_velocity: float = 0.0  # MOM6 DRAG_BG_VEL [m/s]; when >0,
                                           # drag is quadratic-with-floor:
                                           # tau ∝ √(u²+v²+u_bg²) · u, with
                                           # the linear-in-u limit set to
                                           # bottom_drag_r at |u|→0.

    # --- Tracer diffusivity & vertical mixing (A_v/K_v are constant fallbacks
    #     unless a physics vertical-mixing scheme / implicit_vertical_mixing
    #     overrides them) ---
    K_h: float = 0.0
    K_bih: float = 0.0
    A_v: float = 1.0e-3
    K_v: float = 1.0e-4

    # === Chronological (positional-stability) tail — grouped in the docstring
    #     map, NOT by field order: barotropic solver, conservation, numerics
    #     choices, runtime-check bounds, polar-cap boost, EOS/physics, constants.
    n_barotropic_substeps: int = 30
    hyperdiff_coeff: float = 0.0
    use_conservation_fixer: bool = False
    fix_volume: bool = True
    fix_heat: bool = True
    fix_salt: bool = True
    # Issue #271: standalone end-of-step volume-drift projection that
    # runs independently of ``use_conservation_fixer``.  The lat-lon
    # C-grid path leaks ~0.4 mm/yr of mean eta with ETOPO bathymetry
    # (0.06 mm/yr flat) because the partial-cell face masking creates a
    # small mismatch between the depth-integrated tracer transport and
    # the barotropic ``Hu_avg``.  This projection forces
    # ``sum(eta_new * area) == sum(eta_old * area) + dt * sum(F_eta * area)``
    # exactly each step, identical in spirit to MOM6/NEMO/MITgcm
    # practice.  Default-on for lat-lon C-grid; MPAS already conserves
    # to machine precision.
    fix_eta_drift: bool = True
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
    tracer_advection: str = "tvd"  # "upwind", "centered" (unlimited 2nd-order, Veros adv_flux_2nd), "tvd" (Van Leer), "superbee" (Sweby/Veros), "ppm_fct", "ppm", "dst3", "dst3_multidim", "som", "weno5", "weno7"
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
    # Rigid-lid streamfunction solver knobs (only used when
    # ``barotropic_solver = 'rigid_lid'``).  The rigid lid removes the free
    # surface entirely: the depth-integrated flow is non-divergent and carried
    # by a barotropic streamfunction ψ on vertex (corner) points, solved each
    # step from the elliptic vorticity equation ∇·((1/H)∇)ψ = curl((1/H)∫F dz)
    # (Veros core/external/solve_stream.py).  The column depth H is FIXED at the
    # bathymetry (no eta dependence).  ψ is integrated with Adams-Bashforth-2
    # reusing ``ab2_epsilon`` as the Veros AB_eps.  The elliptic solve is the
    # AD-safe ``jax.scipy.sparse.linalg.cg`` (the operator is symmetric).  The
    # net transport through multiply-connected/periodic-channel domains is set
    # by the island line-integral constraints (see rigid_lid_islands.py).
    rigid_lid_cg_tol: float = 1.0e-11
    rigid_lid_cg_maxiter: int = 1000
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
    ab2_epsilon: float = 0.1  # AB2 stabilization (MITgcm ABepsBar) — also the
    #   Adams-Bashforth ε for the OUTER integrator (Veros AB_eps=0.1).
    # Outer (baroclinic) time integrator. "forward_euler" (default) = the existing
    # single-step split-explicit scheme. "ab2" = Adams-Bashforth-2 on the EXPLICIT
    # tendency with implicit vertical mixing applied ONCE afterward (Veros-faithful;
    # core/thermodynamics.py tracers + core/external/solve_stream.py momentum):
    #   X*      = X^n + (1.5+ε)·ΔX_expl^n − (0.5+ε)·ΔX_expl^{n-1}
    #   X^{n+1} = ImplicitVertMix(X*)
    # where ΔX_expl is the explicit-only forward-Euler increment (NOT including the
    # implicit vertical mixing), carried on ``{T,S,u,v}_incr_prev``. The barotropic
    # mode is kept from the barotropic solve (un-AB2'd); only the baroclinic momentum
    # deviation is AB2'd. Because implicit vertical mixing is applied once (a
    # backward-Euler solve), the scheme is UNCONDITIONALLY stable in the vertical and
    # compatible with convective adjustment. Do NOT combine with
    # ``tracer_time_integrator="ab2"`` (double-AB2 of the explicit tracer tendency).
    # CONSERVATION: AB2 extrapolates the tracer CONCENTRATION (like Veros), so the
    # area·thickness-weighted heat/salt content is conserved EXACTLY only with fixed
    # layer thickness — ``barotropic_solver="rigid_lid"`` (the faithful ACC config) or
    # ``use_conservation_fixer=True``. Under a moving free surface it has a small
    # O(Δη) tracer-content drift; "forward_euler" conserves to machine zero.
    outer_integrator: str = "forward_euler"
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

    # --- Polar-cap viscosity boost (tripolar fold support) ---
    # Appended at the end of the NamedTuple to preserve positional
    # construction semantics for legacy callers.  When > 1, multiplies
    # A_h by 1 + (boost − 1) · S(|lat| − cap_lat_deg) where S is a
    # smooth tanh ramp of width ``A_h_cap_width_deg``.  Damps the
    # bipolar-cap cascade on tripolar grids where the cos(lat) scaling
    # drops to zero at the fold boundary but the deformed cap cells
    # need stronger dissipation than ``A_h_floor`` alone provides.
    # Typical ORCA1 production: 5–20.  Disabled by default (1.0) to
    # preserve bit-exact regression on legacy lat-lon configs.
    A_h_cap_boost: float = 1.0
    # Latitude (°N) at which the polar-cap boost ramp begins.  For
    # tripolar grids, set close to the ``fold_lat`` of the
    # FoldDescriptor.  Typical 70–80°.
    A_h_cap_lat_deg: float = 75.0
    # Half-width of the polar-cap boost tanh transition [°]; default 5°.
    A_h_cap_width_deg: float = 5.0
    # Ocean-scoped physical constants (Phase G, G-C1). Defaults reference
    # legoesm.constants (canonical Earth) -> zero behaviour change. A recipe
    # pins these to a reference model (e.g. Veros) via the public config API.
    # Read-through wiring (de-mirroring) is G-C2+. NOTE: no field whose default
    # READS the `constants` module may be declared after this one — the default
    # here assigns the class-body name `constants` to a ConstantsConfig
    # instance, shadowing the module. Fields with literal defaults (e.g.
    # momentum_flux_scheme below) are fine to append.
    constants: ConstantsConfig = ConstantsConfig()
    # Horizontal momentum-flux reconstruction, used ONLY when
    # momentum_advection="flux_form": "upwind" (1st-order, dissipative, stable)
    # or "centered" (2nd-order, non-dissipative). Ignored by the
    # vector_invariant / weno momentum paths. Literal default -> safe after
    # `constants`.
    momentum_flux_scheme: str = "upwind"
    # Lateral (harmonic) momentum-viscosity OPERATOR form. Selects how the A_h
    # Laplacian viscosity acts on the vector velocity field:
    #   "vector_laplacian" (default) — legoESM's VECTOR Laplacian
    #     ∇²_vec(u,v) = grad(div) − k×grad(curl) (``vector_laplacian_cgrid``),
    #     which carries the spherical curvature coupling between u and v. The
    #     paired K_diss_h (when source_kdiss_h+kdiss_h_flux_form) is the Helmholtz
    #     form A_h·(div²+ζ²) (``vector_laplacian_dissipation_cgrid``). DEFAULT ⇒
    #     every existing run is BIT-IDENTICAL.
    #   "flux_divergence" — Veros's component-wise FLUX-DIVERGENCE harmonic
    #     friction ∇·(A_h∇u), ∇·(A_h∇v) per velocity component, with NO
    #     curvature coupling (``flux_divergence_viscosity_cgrid``; matches
    #     ``veros/core/friction.py`` ``harmonic_friction``). The same cos(lat)
    #     A_h scaling (``A_h_lat_scaling``/``A_h_cos_power``) is applied INSIDE the
    #     flux as Veros's ``enable_hor_friction_cos_scaling``/``hor_friction_cosPower``.
    #     The paired K_diss_h is the energy-consistent component-wise A_h·|∇u|²
    #     (Veros ``calc_diss_u``/``calc_diss_v``), built from the SAME face fluxes.
    #     B_h biharmonic / Smagorinsky / Leith are unaffected (they still use the
    #     vector operators). The ACC recipe opts in. Literal default -> safe after
    #     `constants`.
    lateral_viscosity_operator: str = "vector_laplacian"
    # Asynchronous ("distorted-physics") time stepping: dt_mom = dt / dt_mom_ratio.
    # The `dt` passed to step()/integrate_scan IS dt_tracer (the clock — Veros
    # advances vs.time by dt_tracer), and momentum + the barotropic solve + implicit
    # vertical FRICTION are integrated with the SHORTER dt_mom, while tracers +
    # continuity/eta + implicit vertical DIFFUSION + the clock use dt_tracer. This is
    # Veros's dt_mom≠dt_tracer (acc.py dt_mom=4800, dt_tracer=43200 ⇒ ratio 9): NOT a
    # subcycle (momentum() runs once), an under-relaxation that accelerates the
    # transient to the SAME steady state. Default 1.0 ⇒ dt_mom == dt_tracer == dt ⇒
    # BIT-IDENTICAL. Requires barotropic_solver="rigid_lid" when != 1.0: under the
    # rigid lid the column depth H is fixed, so the tracer flux-form update (h fixed)
    # is exactly dt-independent and tracer mass is conserved; a moving free surface
    # would mix a dt_mom-evolved thickness with a dt_tracer flux divergence and leak
    # O((dt_tracer−dt_mom)·∂h/∂t) tracer mass (rejected at config validation).
    dt_mom_ratio: float = 1.0
    # Apply the surface TRACER forcing (T*/S* restoring + prescribed q_net +
    # penetrating shortwave) IMPLICITLY inside the backward-Euler vertical-mixing
    # solve — matching Veros, which adds ``dt_tracer·forc/dz[surface]`` to the
    # implicit vertical-diffusion tridiagonal RHS at weight 1.0
    # (``veros/core/thermodynamics.py``), NOT as an AB2-extrapolated explicit
    # tendency. When True, that surface forcing is WITHHELD from the explicit
    # ``dT_dt``/``dS_dt`` (so under the faithful AB2 outer integrator it is not
    # over-applied by the 1.6× extrapolation) and routed into
    # ``LatLonCGridOceanTendencies.surface_tracer_forcing``, which the model step
    # adds (× dt_tracer) to the tracer solve INPUT before the tridiagonal solve.
    # WIND STRESS (→ du_dt/dv_dt) is unaffected — it is explicit/AB2'd in BOTH
    # legoESM and Veros and already matches. Requires
    # ``implicit_vertical_mixing=True`` (the implicit solve is where the source
    # is placed); rejected otherwise at config validation. Default False ⇒
    # current EXPLICIT surface-forcing placement ⇒ BIT-IDENTICAL.
    surface_forcing_implicit: bool = False

    # --- Adaptive-implicit vertical momentum advection ---
    # (Shchepetkin 2015 / NEMO ``ln_zad_Aimp``).  Appended at the end of
    # the NamedTuple to preserve positional construction for legacy
    # callers.  When True, the explicit first-order-upwind vertical
    # momentum advection (which has no vertical-CFL limit and amplifies a
    # spurious ``w`` super-exponentially in thin cells -- the documented
    # OMIP cold-start "vertadv" runaway) is replaced by a Courant-split
    # explicit/backward-Euler-implicit scheme that is unconditionally
    # stable, monotone, and conservative.  The split uses the
    # barotropic-consistent ``w`` at the step level, so when enabled the
    # vertical momentum advection is applied AFTER the barotropic solve
    # (and the in-tendency explicit ``vertadv`` term is skipped).
    # Disabled by default (False) to preserve bit-exact regression on
    # legacy lat-lon configs; real eORCA/OMIP production runs enable it.
    adaptive_implicit_vertadv: bool = False

    # --- Outer baroclinic momentum time integrator ---
    # "euler" (default): single-step forward-Euler of the momentum
    #   perturbation tendency (legacy; bit-exact).
    # "rk3": 3-stage SSP-RK3 (Shu-Osher) of the momentum perturbation
    #   tendency, mirroring NEMO's RK3 outer step (compile-time key_RK3 in
    #   ORCA1).  Forward-Euler has no stability region for the advective /
    #   relative-vorticity / pressure-gradient terms, so the violent
    #   cold-start geostrophic adjustment from rest amplifies; RK3's
    #   stability region (|z| up to ~sqrt(3) on the imaginary axis) carries
    #   it.  T,S,eta + surface forcing are frozen across the 3 stages
    #   (operator-split with the Matsuno Coriolis + barotropic + tracer
    #   stages); the barotropic slow forcing is taken from stage 1.  3x the
    #   tendency cost.  Appended at the END of the NamedTuple to preserve
    #   positional construction for legacy callers.
    momentum_time_integrator: str = "euler"

    # --- Surface freezing-point floor (sea-ice thermodynamic surrogate) ---
    # When True, ocean temperature is floored at ``freeze_floor_temp_c`` [degC]
    # at the END of each step.  legoESM has no prognostic sea ice, so an
    # exposed high-latitude cell super-cools several degrees below the freezing
    # point of seawater — physically impossible (ice would form, latent heat
    # holding SST at freezing) and 3-5 C colder than NEMO, whose LIM sea ice
    # caps SST at the freezing point.  This floor is that thermodynamic cap:
    # the same ``jnp.maximum(T, T_freeze)`` clamp the slab oceans already apply
    # (simple_ocean.py).  The implied freeze (latent) heat is NOT carried as an
    # ice tracer, so this is a bounded, justified non-conservation representing
    # ice formation (validated: it removes the ~0.5 C global / ~half the Arctic
    # SST RMSE vs NEMO on the eORCA025 CORE-II run).  Off by default to keep
    # bit-exact regression on legacy configs; enabled for faithful OMIP runs.
    freeze_floor: bool = False
    # Freezing point of seawater in degC (model T is in degC).  Defaults to
    # ``T_freeze_ocean - T_freeze`` = -1.8 C (constants, not a literal).
    freeze_floor_temp_c: float = _T_FREEZE_OCEAN_C
    # --- Fourier polar filter (lat-lon pole CFL stabiliser) ----------------
    # A global lat-lon ocean has converging meridians: dx = R*dlon*cos(lat) -> 0
    # at the poles, so explicit advection/metric terms violate CFL near the pole
    # and the cold-start blows up (~day 0.25) regardless of integrator.  When
    # enabled, ``LatLonCGridOceanModel`` truncates the zonal Fourier modes that
    # exceed the per-latitude CFL limit poleward of the cutoff (the existing
    # ``grids.polar_filter``, already used by the atmosphere C-grid).  The filter
    # is MASK-AWARE: land cells are filled with the per-latitude ocean zonal mean
    # before the FFT and restored afterward, so continental zeros are not smeared
    # into adjacent ocean (a naive zonal FFT would couple basins across land).
    # The per-latitude WET-CELL zonal mean is restored after filtering: conserves
    # the per-row ocean volume (eta) / zonal-mean flow (u) exactly, and tracer
    # content exactly only where layer thickness is zonally uniform (approximately
    # under partial cells / z*; a stability filter, not a flux operator).
    # Off by default (bit-exact for tripole/regression configs).  NOTE: even
    # mask-aware, a lat-lon grid cannot be fully faithful in the land-locked
    # Arctic (basins still couple weakly across the pole) — that is why the
    # faithful OMIP path uses the ORCA tripole; this is for the lat-lon grid's
    # own stability + a tropics/mid-lat/SH comparison.
    use_polar_filter: bool = False
    polar_filter_cutoff_lat_deg: float = 60.0
    # Max wave speed [m/s] setting the per-latitude CFL wavenumber cap (external
    # gravity wave ~200-300 m/s; larger -> more aggressive truncation).
    polar_filter_max_wave_speed: float = 300.0
    # Fraction of the theoretical CFL wavenumber kept (<1 for margin).
    polar_filter_safety_factor: float = 0.85
