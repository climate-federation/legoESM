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
    chl : array or None
        Surface chlorophyll [mg/m³] for the RGB shortwave-penetration scheme
        (``ShortwavePenetrationConfig.scheme == "rgb_chl"``).  2D horizontal
        field; ``None`` when the two-band Jerlov scheme is in use.
    q_prescribed : array or None
        Prescribed part of the surface heat flux [W/m², positive into ocean]
        consumed ONLY by the ``"flux_feedback"`` surface-forcing scheme
        (Veros global_4deg ``qnet``).  Kept separate from ``q_net`` so the
        feedback scheme owns the TOTAL heat in one place (ice mask) and so
        the prescribed-channel ``c_sw`` seam is not double-counted — leave
        ``q_net=None`` when using ``flux_feedback``.
    q_feedback : array or None
        Linear SST-feedback (piston) coefficient [W/m²/K, ≥ 0 damps] for the
        ``"flux_feedback"`` scheme (Veros ``qnec``).  Heat contribution is
        ``q_feedback · (T_feedback_target − T_surf)``.
    T_feedback_target : array or None
        Target SST [°C] for the ``q_feedback`` term (Veros ``sst_clim``,
        monthly-interpolated by the driver/harness).
    S_restore_target : array or None
        Target SSS [PSU] for the ``flux_feedback`` scheme's surface-salinity
        restoring (Veros ``sss_clim``).
    q_solar : array or None
        Penetrative SOLAR component of the surface heat flux [W/m², positive
        into ocean] consumed ONLY by the ``"flux_feedback"`` scheme when
        ``FluxFeedbackConfig.penetrative_shortwave=True`` (Veros
        global_flexible / global_1deg ``qsol``).  Deposited through the water
        column with the shared two-band Jerlov profile
        (``shortwave_penetration_tendency``, water type from
        ``FluxFeedbackConfig.shortwave_water_type``; type "I" = exactly the
        Veros literals R=0.58, ζ1=0.35 m, ζ2=23.0 m).

        HEAT-OWNERSHIP CONTRACT (no double counting): when ``q_solar`` is
        provided, ``q_prescribed`` must carry the NON-SOLAR remainder only
        (harness: ``q_prescribed = qnet_total − qsol``).  legoESM deposits
        100% of ``q_solar`` in the column with the I(0)=1 surface convention;
        Veros instead keeps the solar-inclusive total in ``qnet`` and applies
        a zero-column-sum redistribution built with pen(0)=0 — the two are
        algebraically identical cell by cell (top cell receives
        ``qnet_total − qsol·I(z₁)`` either way, deeper cells receive the same
        interface-flux differences, and both catch the residual light in the
        deepest grid cell).  The simple ice mask is evaluated on the TOTAL
        flux ``q_prescribed + feedback + q_solar`` (= Veros's
        ``forc_temp_surface`` whose qnet includes solar) and zeroes the
        surface deposit AND the full solar column (Veros ``ice[..., None]``).
        Mixing ``q_solar`` with the ``sw_down``/``q_net`` channels raises at
        trace time.
    S_restore_piston : array or None
        Per-cell PISTON VELOCITY [m/s] for the ``flux_feedback`` scheme's
        surface-salinity restoring (EXT-N3; Veros north_atlantic ``sss_rest``
        = forcing-file field / 100, ``north_atlantic.py:245-249`` — per-cell,
        monthly; interpolation to the current time is driver/harness work,
        this channel is per-step traced).  When given (requires
        ``FluxFeedbackConfig.salt_restore_piston=True`` and
        ``S_restore_target``), the restoring rate becomes

            dS/dt = S_restore_piston · (S_restore_target − S_surf) / dz₀

        (Veros ``forc_salt_surface = sss_rest·(sss_clim − S)·maskT``
        [PSU·m/s], ``north_atlantic.py:336-340``, divided by the top-cell
        thickness in the implicit RHS, ``core/thermodynamics.py:282``),
        REPLACING the scalar ``1/tau_restore_s`` form.  Passing it without
        the config gate (or relying on the scalar while the gate is on)
        raises at trace time — never a silent fallback.  The ice mask zeroes
        this rate exactly like the scalar form (Veros zeroes
        forc_salt_surface, ``north_atlantic.py:342-344``).
    """
    sw_down: object = None       # jnp.ndarray | None
    q_net: object = None         # jnp.ndarray | None
    tau_x: object = None         # jnp.ndarray | None
    tau_y: object = None         # jnp.ndarray | None
    freshwater: object = None    # jnp.ndarray | None
    salt_flux: object = None     # jnp.ndarray | None  (real salt mass, kg/m2/s)
    chl: object = None           # jnp.ndarray | None  (surface chlorophyll, mg/m3)
    # --- "flux_feedback" scheme channels (None ⇒ inert; see docstring) ---
    q_prescribed: object = None        # jnp.ndarray | None  [W/m²]
    q_feedback: object = None          # jnp.ndarray | None  [W/m²/K]
    T_feedback_target: object = None   # jnp.ndarray | None  [°C]
    S_restore_target: object = None    # jnp.ndarray | None  [PSU]
    q_solar: object = None             # jnp.ndarray | None  [W/m²] (penetrative)
    S_restore_piston: object = None    # jnp.ndarray | None  [m/s] (per-cell SSS piston)


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
    # NEMO zdfdrg drag laws (mirror of DynBottomDragConfig; see there):
    # "legacy" keeps the r/bg_velocity path bit-exact; "nemo_quadratic" /
    # "nemo_loglayer" compute r = Cd·√(u²+v²+ke0) from the bottom-cell
    # speed (loglayer: Cd = clip((κ/ln(½h_bot/z0))², cd0, cdmax)).
    bottom_drag_scheme: str = "legacy"
    bottom_drag_cd0: float = 1.0e-3     # NEMO rn_Cd0 [-]
    bottom_drag_cdmax: float = 0.1      # NEMO rn_Cdmax [-]
    bottom_drag_z0: float = 3.0e-3      # NEMO rn_z0 [m]
    bottom_drag_ke0: float = 2.5e-3     # NEMO rn_ke0 [m²/s²]
    # smc03 PGF: use the 3-point 2nd-order backward bottom-cell density slope
    # (curvature-accurate under a pressure-dependent EOS) instead of the
    # O(dz)-biased one-sided slope.  Default False keeps the proven smc03 path
    # bit-exact; opt-in for the cubed-sphere cold-start over steep partial cells.
    smc03_bottom_2nd_order: bool = False
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
    # Prognostic turbulent kinetic energy [m^2/s^2] at the interior interfaces
    # (W-grid), 3-D Field (n_lat, n_lon, nlev-1), used only when the prognostic
    # TKE vertical-mixing closure is active (vertical_mixing.tke.prognostic=True).
    # Carried across model steps: each step runs ONE backward-Euler TKE solve
    # (dt = dt_mom) seeded from this field and stores the updated TKE back.
    # Default None -> inert (Mode-B diagnostic chain): zero behaviour change.
    tke: object = None
    # Prior-step ADVECTIVE TKE tendency [m^2/s^3] at the interior interfaces
    # (W-grid), 3-D Field (n_lat, n_lon, nlev-1) — the Adams-Bashforth history
    # dtke^{n-1} for the prognostic-TKE superbee advection (Veros vs.dtke,
    # tke.py:292-323), used only when vertical_mixing.tke.advection_scheme !=
    # "none" (requires prognostic=True). None on the first step => the AB2
    # increment uses a zero previous tendency, exactly like Veros's
    # zero-initialised dtke[taum1]. Default None -> inert: zero behaviour change.
    dtke: object = None
    # Carried EKE dissipation rate [m^2/s^3] at the interior interfaces (W-grid),
    # 3-D Field (n_lat, n_lon, nlev-1). Populated by the 3-D EKE step (Veros
    # eke_diss_iw, run in the GM/Redi stage) and consumed WITHIN THE SAME step by
    # the prognostic TKE source when vertical_mixing.tke.source_eke_diss=True
    # (the TKE source reads state_new.eke_diss — this step's EKE update — since
    # the implicit-vmix TKE solve runs after GM/Redi, matching Veros's same-step
    # eke->tke ordering). The CARRIED value on state is only the fallback when a
    # step produces none (e.g. the 2-D EKE path). Default None -> inert: zero
    # behaviour change.
    eke_diss: object = None
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
    # Previous-step barotropic slow forcing (depth-mean tendency) for the
    # AB2 time-centering of F_slow (matches Oceananigans' AB2-extrapolated Gᵁ).
    # Only used when barotropic_slow_forcing_ab2=True; None otherwise (default).
    F_slow_u_prev: object = None
    F_slow_v_prev: object = None
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

    The same (dT_dt, dS_dt) rate-pair container is REUSED for the implicit
    COLUMN tracer source on the SEPARATE
    ``LatLonCGridOceanTendencies.tracer_source`` slot
    (``sponge_forcing_implicit``, the Veros ``tempsalt_sources`` placement).
    The slots must stay distinct: the post-mixing TKE surface buoyancy-flux
    reconstruction column-sums THIS slot to rebuild Veros's
    ``forc_temp_surface`` and must exclude column sources (Veros keeps
    tempsalt_sources out of forc_rho_surface).

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
    # Bottom-drag KE-extraction dissipation density [m²/s³] at the interior
    # interfaces (W-grid, (n_lat, n_lon, nlev-1)) — Veros K_diss_bot. Populated
    # ONLY when the prognostic-TKE ``source_bottom_drag_diss`` option is on (the
    # ACC recipe), so the prognostic TKE source can recycle the bottom-drag KE
    # extraction (Veros integrate_tke ``forc += K_diss_bot``). ``None`` otherwise
    # (default), keeping the tendency pytree + every existing path bit-identical.
    K_diss_bot: object = None
    # --- AB2 "advective"-scope dissipative split (Veros-faithful) ---
    # When ``ab2_scope="advective"`` the DISSIPATIVE momentum / tracer
    # tendencies are WITHHELD from ``du_dt``/``dv_dt`` / ``dT_dt``/``dS_dt``
    # (so the outer AB2 extrapolates only the ADVECTIVE part) and exposed here
    # so the model step can apply them at WEIGHT 1.0 (forward-Euler), matching
    # Veros's placement (momentum friction + bottom drag in
    # ``core/external/solve_stream.py``; tracer lateral diffusion in
    # ``core/thermodynamics.py``). ``du_diss``/``dv_diss`` carry the lateral
    # friction (whichever ``lateral_viscosity_operator``) + bottom-drag
    # momentum tendencies [m/s²]; ``dT_diss``/``dS_diss`` carry the lateral
    # tracer-diffusion tendency [degC/s, PSU/s] (the GM/Redi isoneutral+skew
    # part is added to these in the model step, where it is computed). ``None``
    # otherwise (default ``ab2_scope="total"``), keeping the tendency pytree +
    # every existing path bit-identical.
    du_diss: object = None
    dv_diss: object = None
    dT_diss: object = None
    dS_diss: object = None
    # --- Implicit COLUMN tracer source (Veros tempsalt_sources placement) ---
    # tracer_source is a :class:`SurfaceTracerForcing`-typed (dT_dt, dS_dt)
    # full-column RATE pair, populated ONLY when
    # ``LatLonCGridOceanConfig.sponge_forcing_implicit`` is on: the sponge T/S
    # relaxation rates are then WITHHELD from ``dT_dt``/``dS_dt`` (so the AB2
    # outer integrator never extrapolates them) and applied at weight 1.0
    # inside the backward-Euler implicit vertical-mixing solve — Veros's
    # ``tempsalt_sources`` placement (forward-Euler at taup1 BEFORE the
    # implicit vmix; ``veros/core/thermodynamics.py:419`` →
    # ``veros/core/diffusion.py:132-141``).  Carried on a SEPARATE slot from
    # ``surface_tracer_forcing`` because the post-mixing TKE surface
    # buoyancy-flux reconstruction column-sums ``surface_tracer_forcing`` to
    # rebuild Veros's ``forc_temp_surface`` — a column SOURCE must stay out of
    # that sum, exactly as Veros keeps ``tempsalt_sources`` out of
    # ``forc_rho_surface`` (thermodynamics.py:312-314).  ``None`` otherwise
    # (default), keeping the tendency pytree + every existing path
    # bit-identical.
    tracer_source: object = None


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
    docs/ocean/experiments/global_overturning_plan.md.

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


class OMp25Config(NamedTuple):
    """OM4p25 lateral-friction closure coefficients (GFDL OM4.0, Adcroft et al.
    2019; Silvestri et al. 2024 "SM2"). Laplacian + biharmonic, each the max of
    a Smagorinsky term and a static grid-scale term, with the Laplacian tapered
    by the deformation-radius factor F = 1/(1+0.25·(L_d/Δ)⁴). Defaults are the
    published OM4p25 values."""
    C2: float = 0.15      # Laplacian Smagorinsky coefficient
    Cu2: float = 0.01     # Laplacian static-viscosity coefficient
    C4: float = 0.06      # biharmonic Smagorinsky coefficient
    Cu4: float = 0.01     # biharmonic static-viscosity coefficient
    deformation_radius_m: float = 6.75e3  # L_d for the F taper [m]; ~uniform for
    #   the idealised baroclinic jet. Spatially-varying L_d (from N²) is a refinement.


class DynBottomDragConfig(NamedTuple):
    """Dynamics-level bottom-drag parameters (#501 config grouping).

    Distinct from the deprecated physics-pathway ``BottomDragConfig``.  Field
    names retain the ``bottom_drag_`` prefix so the flat YAML / legacy-kwarg
    interface maps 1:1 through ``LatLonCGridOceanConfig.from_flat``.
    """

    bottom_drag_r: float = 0.0
    bottom_drag_bbl_thickness: float = 0.0
    bottom_drag_bg_velocity: float = 0.0  # MOM6 DRAG_BG_VEL [m/s]; when >0,
                                           # drag is quadratic-with-floor:
                                           # tau ∝ √(u²+v²+u_bg²) · u, with
                                           # the linear-in-u limit set to
                                           # bottom_drag_r at |u|→0.
    # --- NEMO zdfdrg drag laws (nemo_5.0.1 zdfdrg.F90) ---
    # "legacy" (default) = the historical r/bg_velocity/BBL path above,
    # bit-exact.  "nemo_quadratic" (zdfdrg np_non_lin — the ORCA1
    # namelist selection) and "nemo_loglayer" (np_loglayer) instead
    # compute r = Cd·√(ū²+v̄²+ke0) from the BOTTOM-cell velocity at the
    # tracer point (full speed, background KE in quadrature) with, for
    # the log-layer, Cd = clip((κ/ln(½h_bot/z0))², cd0, cdmax).  The
    # NEMO schemes ignore ``bottom_drag_r``/``bottom_drag_bg_velocity``
    # and are enabled by the scheme string alone;
    # ``bottom_drag_bbl_thickness`` still selects the K&E99 spread
    # (NEMO applies drag to the bottom cell only: set it to 0 for
    # strict NEMO behaviour).
    bottom_drag_scheme: str = "legacy"
    bottom_drag_cd0: float = 1.0e-3     # NEMO rn_Cd0 [-] (loglayer: Cd min)
    bottom_drag_cdmax: float = 0.1      # NEMO rn_Cdmax [-] (loglayer Cd cap)
    bottom_drag_z0: float = 3.0e-3      # NEMO rn_z0 [m] bottom roughness
    bottom_drag_ke0: float = 2.5e-3     # NEMO rn_ke0 [m²/s²] background KE


class BarotropicConfig(NamedTuple):
    """Barotropic free-surface solver parameters (#501 config grouping).

    The split-explicit substep loop, the implicit-CN PCG knobs, and the
    rigid-lid streamfunction-solver knobs.  Field names retain their original
    flat names so the YAML / legacy-kwarg interface maps 1:1 through
    ``LatLonCGridOceanConfig.from_flat`` / ``flat_fields``.
    """

    n_barotropic_substeps: int = 30
    barotropic_diffusion_alpha: float = 0.01
    barotropic_diffusion_dt_ref: float = 60.0
    barotropic_div_damp: float = 0.0  # Divergence damping on barotropic velocity (dimensionless)
    bebt: float = 0.2               # Semi-implicit barotropic PGF [0,1]. 0=forward-backward, 0.2=MOM6 default.
    maxvel_barotropic: float = 0.0  # Velocity clipping [m/s]. 0=disabled. MOM6 uses 6.0.
    barotropic_time_filter: str = "cosine"  # "box", "cosine", or "power_law" (SM2005 ROMS/MOM6/Oceananigans extended-window filter; damps the 2dx barotropic Coriolis null mode)
    differentiable_barotropic: bool = False
    # SOTA-local split-explicit barotropic (MOM6/MPAS-Ocean style): when True the
    # per-substep eta-floor clamp is LOCAL (jnp.maximum, NO allreduce) and the
    # global mass-conserving redistribute runs ONCE per outer barotropic step on
    # the time-averaged eta, instead of EVERY substep.  clamp_and_redistribute
    # does n_iter=3 batched allreduces/call, so this cuts the barotropic SUBCYCLE
    # from ~3*n_substeps allreduces/step (e.g. 90 at n_substeps=30; doubled if
    # barotropic diffusion is active) to 3 -> a halo-only subcycle = the
    # multi-node strong-scaling lever (the implicit_cn analogue is the
    # 120-allreduce PCG wall).  NOTE: the outer-step fix_eta_drift fixer is a
    # SEPARATE reduction, unaffected by this flag.  BIT-IDENTICAL to the
    # per-substep-redistribute path whenever no cell hits eta_floor (deep ocean,
    # no wetting/drying), since both the local jnp.maximum and the redistribute
    # are then no-ops; differs only in active wetting/drying, where per-step (not
    # per-substep) global mass correction is the SOTA-standard approximation
    # (loses per-substep far-field sea-level compensation).  Default False.
    barotropic_local_subcycle_clamp: bool = False
    # AB2 time-centering of the barotropic slow forcing F_slow (matches the
    # Oceananigans split-explicit Gᵁ = AB2-extrapolated depth-integral of the 3D
    # tendency, vs legoESM's default current-time depth-mean).  Investigated for
    # the §5 no-in-substep-Coriolis path's barotropic geostrophic balance
    # (docs/dev-notes/issues/barotropic_mode_noise.md). Default False = bit-identical.
    barotropic_slow_forcing_ab2: bool = False
    # Barotropic solver selection (see docs/dev-notes/issues/barotropic_mode_noise.md).
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
    # Distributed (MPI) implicit-CN knobs.  Under MPI the stock
    # ``jax.scipy`` CG deadlocks (rank-local dot products + a residual-
    # dependent ``while_loop`` desynchronise the collective schedule), so
    # the multi-rank path runs a HAND-ROLLED fixed-iteration PCG of
    # exactly ``barotropic_implicit_pcg_fixed_iters`` iterations (static
    # ``fori_loop`` => uniform collective schedule, no deadlock),
    # UNROLLED and differentiated straight through (the halo
    # ``_sendrecv_vjp`` + the ``allreduce(SUM)`` dots are AD-safe;
    # ``custom_linear_solve`` is NOT used because it cannot transpose the
    # MPI-halo ``custom_vjp`` — see barotropic_common's module note).  The
    # single-rank path is UNCHANGED (still stock CG).  Default 60 is a
    # conservative estimate for 1e-10 residual on a diagonally-dominant
    # Helmholtz at 1°-¼°; it MUST be validated against the returned global
    # residual (``barotropic_implicit_pcg_residual_tol``) for each deck —
    # tripole-fold / coastal conditioning can require more.  See
    # docs/ocean/experiments/distributed_barotropic_pcg.md.
    barotropic_implicit_pcg_fixed_iters: int = 60
    barotropic_implicit_pcg_residual_tol: float = 1.0e-10
    # Force the fixed-iteration PCG even when not distributed.  Two uses:
    # (1) solver-matched serial parity references — the np>=2 implicit_cn
    # path ALWAYS runs the fixed-M PCG, so a serial reference using stock
    # CG differs at the solver-residual level by construction (parity
    # bisect job 8459362: identical "MISMATCH" across all halo knobs);
    # (2) performance — the fixed-M PCG measured FASTER single-rank than
    # stock CG (20.8 vs 24.6 ms, job 8458701).  Under a single process
    # the PCG's global dots are plain local sums (``is_multi_process()``
    # gate), so this is safe pre-arming inside an MPI job.
    barotropic_implicit_force_pcg: bool = False
    # Distributed-PCG body variant (only used on the fixed-M PCG path):
    #   "standard"      — 2 sequentially-dependent reductions/iter.
    #   "single_reduce" — Chronopoulos-Gear recurrences, ONE batched
    #                     reduction/iter (M+1 vs 2M+1 per solve) — the
    #                     multi-node weak-scaling lever (probe 8460255:
    #                     np32 weak growth was allreduce-latency-bound).
    # Equivalent in exact arithmetic; differs at round-off (solver-
    # tolerance lane, not bit-exact).  Validated at solver dispatch
    # (unknown ⇒ ValueError).  OPT-IN, NOT a default (regime-dependent,
    # measured): single_reduce wins ONLY when the barotropic reductions
    # dominate the step — small per-rank tiles / high rank counts /
    # multi-node (LL12 rows/rank: +3-7% np2/4, job 8470723).  At a
    # PRODUCTION tile (rows/rank=48, job 8475875) the barotropic solve
    # is ~6-7% of the step (vmix dominates), so the variant is
    # within-noise neutral — NOT worth flipping the default and risking
    # the bit-repro lane.  Set it per deck when reduction-latency-bound.
    barotropic_implicit_pcg_variant: str = "standard"
    # Preconditioner for the implicit-CN Helmholtz PCG (validated at the
    # solver entry; unknown ⇒ ValueError):
    #   "jacobi"     — inverse diagonal (legacy default).
    #   "zonal_line" — exact periodic-tridiagonal solves per latitude
    #                  row (cyclic Thomas).  COMMUNICATION-FREE under
    #                  band MPI (each rank owns full longitude rows) and
    #                  inverts exactly the pole-tightened zonal
    #                  couplings that dominate the lat-lon condition
    #                  number — the EVP-block-preconditioner principle
    #                  (CESM POP, GMD 9:4209: fewer latency-bound
    #                  iterations for cheap local FLOPs).  W-self-adjoint
    #                  by construction (single_reduce-compatible).
    # OPT-IN, regime-dependent (measured, same caveat as the variant
    # above): zonal_line + M=20 is +20-26% at small/reduction-bound
    # tiles (LL12 ≤8/node, job 8475325) but ~neutral at a production
    # tile (rows/rank=48, job 8475875) where the barotropic solve is
    # only ~6-7% of the step AND the cyclic-Thomas's sequential
    # per-iteration FLOPs offset the M=60→20 iteration cut when
    # compute-bound.  Use it on reduction-latency-bound decks; the
    # default stays "jacobi".
    barotropic_implicit_preconditioner: str = "jacobi"
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
    # Wide-halo split-explicit barotropic (scaling-audit item 3; only used
    # when ``barotropic_solver = 'explicit_substep'``).  When True and a
    # lat-band decomposition is active, the substep loop exchanges ONE wide
    # halo (width = substeps-per-exchange x per-substep stencil reach) and
    # runs the substeps communication-free on the extended band, instead of
    # ~4 halo pads per substep — the latency lever at >=16 ranks (audit:
    # net-NEGATIVE at 2-GPU scale, where bandwidth beats message count; keep
    # it opt-in, flip per deck only with an A/B receipt).  Serial results are
    # value-identical; per-substep eta clamping runs in the LOCAL mode on
    # this path (see barotropic_substeps_wide_halo_latlon_cgrid's contract).
    barotropic_wide_halo: bool = False
    # Substeps per wide exchange (0 = auto: as many as the local band height
    # allows, i.e. floor(n_lat_local / stencil_reach), capped at the loop
    # length).  With strongly UNEVEN bands (--wet-balance) set this so
    # ``chunk x reach <= min band height`` across ranks — the halo pulls
    # rows from ONE neighbour only.
    barotropic_wide_halo_chunk: int = 0


class RuntimeChecksConfig(NamedTuple):
    """Runtime validity-bound parameters (#501 config grouping).

    The sanity bounds checked by ``_assert_runtime_invariants`` (gated by
    ``enable_runtime_checks``): the maximum |eta| and the tracer (T, S) min/max
    bounds.  Field names are unchanged so the flat YAML / legacy-kwarg interface
    maps 1:1 through ``LatLonCGridOceanConfig.from_flat``.

    ``min_water_column_m`` is deliberately NOT grouped here: it is a physical
    wet-cell thickness floor read by grid-agnostic shared code
    (``ocean_conservation_fixer``, layer-thickness helpers) that also runs on the
    cube ``OceanConfig`` path, so it stays a flat field to keep that read uniform
    across config types.
    """

    enable_runtime_checks: bool = False
    max_abs_eta_m: float = 1.0e4
    temperature_min_c: float = -5.0
    temperature_max_c: float = 45.0
    salinity_min_psu: float = 0.0
    salinity_max_psu: float = 50.0


class LateralViscosityConfig(NamedTuple):
    """Lateral momentum viscosity parameters (#501 config grouping).

    Harmonic (Laplacian) ``A_h`` with latitude / equatorial shaping,
    biharmonic ``B_h``, and the Smagorinsky eddy-viscosity coefficients —
    the lateral momentum dissipation closure for the lat-lon C-grid ocean.
    Field names are unchanged so the flat YAML / legacy-kwarg interface
    maps 1:1 through ``LatLonCGridOceanConfig.from_flat``.  Includes the Leith
    closure (``C_leith``) and the tripole polar-cap A_h boost (``A_h_cap_*``).
    """

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


    # Leith viscosity coefficient (Leith 1996).  When > 0 enables
    # flow-adaptive biharmonic viscosity ``-∇²(A_L ∇²u)`` with
    # ``A_L = (C_L · Δ)³ · |∇ζ|`` (or ``sqrt(|∇ζ|² + |∇δ|²)`` when
    # ``C_leith_modified = True``).  Typical values: 1.0–2.0.  Appended
    # at the END of the NamedTuple so existing positional call sites
    # keep working.
    C_leith: float = 0.0
    C_leith_modified: bool = False

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

class PolarFilterConfig(NamedTuple):
    """Fourier polar-filter parameters (#501 config grouping).

    The lat-lon pole CFL stabiliser (zonal Fourier-mode truncation poleward
    of the cutoff).  Field names unchanged so the flat YAML / legacy-kwarg
    interface maps 1:1 through ``LatLonCGridOceanConfig.from_flat``.
    """

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


# Deferred to break the state <-> physics import cycle: importing
# ``legoesm.ocean.physics.tidal_forcing`` runs ``ocean/physics/__init__`` ->
# ``combined`` -> ``from legoesm.ocean.state import OceanState, OceanSurfaceForcing,
# OceanTendencies``. Those three (the ONLY state symbols the physics package
# imports) are all defined ABOVE, so by this point the cycle resolves cleanly —
# whereas a top-of-file import would fault (state mid-init). Needed at class-def
# time for the ``tidal_forcing`` default below.
from legoesm.ocean.physics.tidal_forcing import TidalForcingConfig  # noqa: E402


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

        cfg = LatLonCGridOceanConfig.from_flat()       # constant A_v/K_v, no physics pipeline

    Production-style::

        cfg = LatLonCGridOceanConfig.from_flat(
            A_h=3e4, A_h_lat_scaling=True, B_h=1e10, C_smag=0.15,
            bottom_drag_r=2.5e-3, implicit_vertical_mixing=True,
            barotropic_solver="implicit", eos="wright",
            physics=OceanPhysicsConfig(...),  # KPP/TKE + GM/Redi
        )

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

    # --- Lateral viscosity (#501 grouped into LateralViscosityConfig) ---
    lateral_viscosity: LateralViscosityConfig = LateralViscosityConfig()

    # --- Bottom drag (dynamics-level; #501 grouped into DynBottomDragConfig;
    #     the physics-pathway BottomDragConfig is deprecated — set drag here) ---
    bottom_drag: DynBottomDragConfig = DynBottomDragConfig()

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
    barotropic: BarotropicConfig = BarotropicConfig()
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
    min_water_column_m: float = 0.5
    runtime_checks: RuntimeChecksConfig = RuntimeChecksConfig()
    freshwater_closure: str = "virtual_salt_flux"
    S_ref: float = 35.0          # Reference salinity for virtual salt flux [PSU]
    # When True, remove the area-mean of the net freshwater flux from the
    # virtual-salt closure so the surface freshwater conserves GLOBAL SALT (the
    # OMIP global freshwater correction; matches the MPAS config field). Default
    # False keeps the legacy raw-flux behaviour bit-exact. Volume is already
    # conserved separately via ``fix_eta_drift``.
    normalize_freshwater: bool = False
    tracer_advection: str = "tvd"  # "upwind", "centered" (unlimited 2nd-order, Veros adv_flux_2nd), "tvd" (Van Leer), "superbee" (Sweby/Veros), "ppm_fct", "ppm", "dst3", "dst3_multidim", "som", "weno5", "weno7"
    gm_redi: object = None         # GMRediConfig or None; enables GM/Redi lateral mixing
    physics: object = None
    eos: str = "wright"
    eos_linear: object = None
    # Slope-foot viscosity enhancement (MOM6 OM4 KH_BG_2D analog).
    # When > 0, multiplies horizontal viscosity (A_h Laplacian, Smagorinsky,
    # Leith) in the bottom N levels by 1 + alpha · tanh(|∇H|/H/δ),
    # locally enhancing dissipation over steep slopes (African shelf,
    # ITF, equatorial trenches). Targets the f≈0 + steep-bathymetry
    # instability mode that constant viscosity cannot reach.
    slope_foot_alpha: float = 0.0       # 0 = disabled; production: 3.0
    slope_foot_threshold: float = 0.1   # MOM6 default
    slope_foot_n_levels: int = 5        # bottom 5 levels
    momentum_advection: str = "vector_invariant"  # "vector_invariant", "weno5", "weno7", "weno9", "flux_form"
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
    # WENO vector-invariant smoothness measure (Silvestri et al. 2024). Selects
    # the "V" vs "D" scheme family for momentum_advection in {weno5,weno7,weno9}:
    #   "split"    (default, = W*V, Oceananigans CrossAndSelfUpwinding): vorticity
    #              uses VELOCITY smoothness {ζ;u} (Eq 43) and divergence uses the
    #              FULL-divergence smoothness {δU; D} (Eq 45). Lower implicit
    #              dissipation / higher effective resolution (the paper's W9V).
    #   "standard" (= W*D, OnlySelfUpwinding): vorticity uses self-smoothness
    #              {ζ;ζ} (Eq 37) and divergence uses self-smoothness {δU; δU}
    #              (Eq 44). The paper notes the divergence choice "has a large
    #              impact on the solution" (W9D is markedly more dissipative).
    weno_smoothness: str = "split"
    # DECOUPLED divergence-flux (D-term) smoothness, independent of the vorticity
    # smoothness above. Oceananigans' WENOVectorInvariant uses TWO independent
    # choices: ``vorticity_stencil`` (VelocityStencil, Eq 43 = our "split" vorticity)
    # AND ``upwinding`` (OnlySelfUpwinding, Eq 44 = "standard"/self divergence) — a
    # mix the single ``weno_smoothness`` flag cannot express ("split" forces the
    # divergence to the more-aggressive Eq-45 full-divergence smoothness; "standard"
    # forces the vorticity to self-smoothness). At marginal (eddy-permitting)
    # resolution the Eq-45 divergence under-dissipates the 2dx grid mode and a
    # baroclinic-eddy field runs away, while Oceananigans (Eq-44 self-divergence)
    # saturates. ``None`` (default) = follow ``weno_smoothness`` for BOTH (bit-
    # identical to the historical behaviour); set ``"standard"`` to get the faithful
    # Oceananigans OnlySelfUpwinding divergence while keeping ``weno_smoothness=
    # "split"`` VelocityStencil vorticity.
    weno_divergence_smoothness: str | None = None
    # Sadourny ENSTROPHY-CONSERVING metric weighting of the vorticity-flux transport
    # velocity. Oceananigans' WENOVectorInvariant builds the transport at the u-point as
    # v̂ = 0.25·Σ(Δx_v·v)/Δx_u (vector_invariant_advection.jl: ℑxᶠᵃᵃ(ℑyᵃᶜᵃ,Δx_q·v)·Δx⁻¹),
    # a Δx (cos-lat) WEIGHTED average — the Sadourny form that conserves enstrophy at
    # FINITE amplitude on a non-uniform metric. legoESM's default forms v_at_u / Fv_at_u
    # as PLAIN 4-point averages (Δx-weighting dropped), which under-dissipates the
    # finite-amplitude 2Δx grid mode (the §5 residual). True selects the faithful
    # metric-weighted transport (and the matching Δy weighting on û for the v-equation);
    # False (default) keeps the plain average (bit-identical to historical behaviour; a
    # no-op on uniform-metric Cartesian grids where Δx_v ≡ Δx_u).
    vortcor_enstrophy_metric: bool = False
    # Reconstruct the RELATIVE VORTICITY ζ directly in the WENO vorticity flux (the
    # Oceananigans WENOVectorInvariant form: flux = v̂·ζᴿ with ζᴿ = WENO(ζ₃ᶠᶠᶜ)), instead
    # of legoESM's default POTENTIAL-vorticity form (reconstruct q=ζ/h, ×mass-flux h·v).
    # The two are identical when h is uniform (η≈0, linear), but at FINITE amplitude η
    # makes h vary and WENO(ζ/h)·(h·v) ≠ WENO(ζ)·v (the WENO is nonlinear over the
    # h-varying stencil) — a candidate for the §5 finite-amplitude under-dissipation.
    # Faithful ONLY on flat-bottom / no-partial-cell setups (the Oceananigans idealized
    # cases): the q-form is retained by default because it conserves potential enstrophy
    # on partial-cell topography (AL81 triad; real ETOPO). False (default) = q-form.
    vortcor_reconstruct_zeta: bool = False
    # WENO vertical momentum advection of the FULL velocity (matches Oceananigans, which
    # advects the full horizontal momentum vertically) instead of legoESM's default
    # baroclinic PERTURBATION u'=u−U_bar. The two differ by the flux-form redistribution
    # −∂(w·U_bar)/∂z = U_bar·∇·u_h (depth-integral zero; U_bar is depth-independent so
    # there is NO extra WENO dissipation, only this redistribution term legoESM omits).
    # Candidate for the INTERIOR finite-amplitude baroclinic-eddy runaway (the §5 residual
    # is at the front, NOT the walls). False (default) = perturbation (bit-identical).
    weno_vertadv_full_velocity: bool = False
    # GH #480: rate [1/s] of the N/S free-slip-wall 2dx-in-lon grid-mode filter,
    # localised to the first/last 8 wall rows (zero in the interior). Default 0.0
    # (OFF). Needed only for eddy-permitting channel runs with WENO vector-invariant
    # momentum + free-slip walls (e.g. the Silvestri §5 jet), where the rotational
    # 2dx wall mode is un-dissipatable by advection (no zonal velocity). NOT a
    # domain viscosity/closure — a boundary Shapiro filter on the wall rows only.
    wall_grid_filter_rate_s: float = 0.0
    # GH #480 (faithful root fix): zero-gradient (Neumann) fill the tracer over
    # land BEFORE the flux-form advection reconstruction, so the wide WENO
    # stencil at the first wet faces sees a flat extension instead of the masked
    # cold land cell (T=0).  The masked cold cell otherwise manufactures a
    # spurious near-wall tracer front that a 2dx-in-lon v perturbation amplifies
    # into an un-dissipatable grid mode at free-slip walls (the §5 eddy-permitting
    # blow-up).  This is the physical no-flux insulating wall = Oceananigans'
    # clean grid-edge wall; the wall-face flux stays zero (mass_flux_u/v), so
    # wet-domain tracer is conserved and interior values are unchanged.  It is a
    # strict no-op where there is no land (periodic/global aquaplanet).  Default
    # ON: the masked cold-cell contamination is a bug for any masked-land run.
    tracer_wall_neumann_fill: bool = True
    # Pressure-gradient force scheme on partial cells.  ``"adcroft"``
    # (default): existing centered-diff p_prime + Adcroft & Campin 2004
    # face-PGF correction.  ``"smc03"``: full Shchepetkin & McWilliams
    # 2003 density-Jacobian PGF with harmonic-mean monotonized slope
    # reconstruction — closes the BH partial-cell gap by avoiding the
    # single-level z-spike that the Adcroft correction produces and that
    # drives the 2Δz computational mode.  See
    # docs/ocean/experiments/density_jacobian_pgf_plan.md.  Pure-z*
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
    # restoring (see docs/ocean/experiments/dino_replication_plan.md
    # Finding 5). Set explicit ``implicit_vertical_mixing=False`` to
    # reproduce the historical explicit-diffusion behavior.
    implicit_vertical_mixing: bool = True

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
    # Stage-8 VERTICAL momentum-advection scheme (independent of the HORIZONTAL
    # momentum_advection dispatch). Selects how -d/dz(w·u) is discretized:
    #   "upwind_perturbation" (DEFAULT, BIT-IDENTICAL) — 1st-order interface
    #     upwind of the BAROCLINIC PERTURBATION u' = u - U_bar. Carries an
    #     implicit vertical viscosity ~|w|·dz/2 that damps baroclinic shear,
    #     and OMITS the depth-integral-zero redistribution term -d/dz(w·U_bar)
    #     (advecting only the perturbation drops the barotropic-momentum part).
    #     Both effects push the column toward barotropic.
    #   "centered_full" (VEROS-FAITHFUL) — 2nd-order CENTERED, energy-conserving
    #     flux of the FULL velocity u (= u' + U_bar), matching the vertical part
    #     of Veros core/momentum.py momentum_advection
    #     (flux_top = 0.25·(u[k+1]+u[k])·(w+w_east)). Restores the w·U_bar
    #     redistribution and removes the upwind implicit viscosity. UNLIMITED ⇒
    #     dispersive (no monotonicity, no implicit viscosity): stability rests on
    #     dt_mom + A_v/TKE friction, like Veros. The ACC recipe opts in.
    # The WENO momentum paths (momentum_advection in {weno5,weno7}) own their own
    # vertical reconstruction and ignore this field. Literal default -> safe
    # after `constants`. Validated at config construction; unknown -> ValueError.
    # REJECTED in combination with adaptive_implicit_vertadv=True (that path
    # replaces the explicit in-tendency vertical advection entirely with an
    # upwind backward-Euler solve, so "centered_full" would be a silent no-op).
    vertical_momentum_scheme: str = "upwind_perturbation"
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
    # Lateral side boundary condition for the harmonic viscosity:
    #   "free_slip" (default) — viscous flux zeroed at walls (∂u_tang/∂n = 0).
    #   "no_slip"  — MITgcm no_slip_sides: adds the wall side-drag
    #                -(2/Δy)·A_h·u (mom_u_sidedrag / mom_v_sidedrag, sideDragFactor=2).
    lateral_side_bc: str = "free_slip"
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
    # --- Fourier polar filter (#501 grouped into PolarFilterConfig) ---
    polar_filter: PolarFilterConfig = PolarFilterConfig()

    # --- Additive momentum vertical-friction placement (Veros) ---
    # Veros computes the implicit vertical-friction increment du_mix from the
    # PRE-STEP velocity u^n (core/friction.py, backward-Euler on u^n) and adds
    # it ADDITIVELY to the AB2-extrapolated explicit tendency
    # (core/external/solve_stream.py: u^{n+1} = u^n + dt·(AB2(du) + du_mix)).
    # legoESM's default placement is SEQUENTIAL: backward-Euler friction on the
    # AB2-advanced state u*.  Both are implicit/unconditionally stable; at
    # equilibrium (AB2(du) ≈ −du_mix) the O(dt²·A_v) placement delta dominates
    # the realized momentum increment (measured: reconstructing the additive
    # form collapses the realized-increment L2 ratio 4.8→1.6 and lifts corr
    # 0.11→0.44 vs Veros — .physics-validator/momentum_fair/).  TRACERS keep
    # the sequential implicit-diffusion-on-the-AB2-state placement in BOTH
    # modes (that IS Veros's tracer placement, core/thermodynamics.py).
    # Requires ``outer_integrator="ab2"`` and ``implicit_vertical_mixing=True``
    # (rejected otherwise at config validation).  The friction solve has
    # zero-flux top/bottom BCs, so the added increment has (thickness-weighted)
    # zero depth-mean and the barotropic mode from the barotropic solver is
    # untouched.  Default False ⇒ sequential placement ⇒ BIT-IDENTICAL.
    momentum_friction_additive: bool = False

    # --- Coriolis time-stepping placement (Veros vs Matsuno split) ---
    # Selects HOW the planetary Coriolis force f×u enters the momentum update:
    #   "matsuno_split" (DEFAULT, BIT-IDENTICAL) — legoESM's existing scheme:
    #     Coriolis is a SEQUENTIAL forward-backward (Matsuno) rotation SUB-STEP
    #     (``_forward_backward_coriolis_3d``) applied to the forward-Euler-advanced
    #     state u* = u^n + dt_mom·du_dt_pert (du_dt EXCLUDES Coriolis), and the
    #     barotropic solver adds its OWN f×u_bt on the barotropic mode. The outer
    #     AB2 then extrapolates the total explicit INCREMENT (which contains the
    #     rotation). The Matsuno one-step map is exactly neutral on the inertial
    #     mode, but AB2-extrapolating its increment numerically DESTROYS
    #     near-inertial energy (|G| 0.65–0.86/step at the ACC channel f·dt_mom).
    #   "explicit_ab2" (VEROS-FAITHFUL) — Coriolis is an EXPLICIT tendency f×u of
    #     the FULL velocity (Veros core/momentum.py tend_coriolisf: the 0.25 C-grid
    #     4-point average of f·v→u-points, −f·u→v-points; legoESM reuses the shared
    #     ``coriolis_cgrid`` operator, which IS that stencil minus two omitted
    #     metric pieces — the tantr curvature term (measured 0.02–0.04% of
    #     Coriolis on the ACC grid) and the meridional dyt·cost/(dyu·cosu)
    #     averaging ratio (up to ~1.6% at the channel edge); BOTH are omitted
    #     identically by the shared C-grid Coriolis machinery on the
    #     matsuno_split path too, so they do not affect the scheme comparison
    #     (adversarial review 2026-06-10, probe_metric.py). It ENTERS ``du_dt``/``dv_dt`` so its depth-mean reaches
    #     the barotropic slow forcing F_slow (= Veros's solve_stream.py uloc/vloc =
    #     depth-integral of du INCLUDING Coriolis) and its perturbation reaches the
    #     3-D du_dt_pert; the outer AB2 extrapolates it with the 1.5/0.6 weights
    #     (Veros AB2-eps). The Matsuno sub-step is SKIPPED and the barotropic
    #     solver's OWN Coriolis addition is GATED OFF (no double count). Per-step
    #     inertial |G| ≈ 0.99–1.01 (weakly anti-damped, like Veros), preserving the
    #     near-inertial / inertia-gravity energy pathway and the discrete-Ekman
    #     angle (the matsuno_split path rotates the Ekman balance ~13–15°).
    # Requires ``outer_integrator="ab2"`` (forward-Euler Coriolis at weight 1.0 is
    # unconditionally UNSTABLE for pure rotation: sqrt(1+(f·dt)²) > 1 per step) AND
    # ``barotropic_solver="rigid_lid"`` (the only barotropic path whose Coriolis IS
    # the depth-mean of the slow forcing; the substep / implicit-CN free-surface
    # solvers sub-step the barotropic Coriolis on the barotropic gravity-wave clock
    # — different physics, out of scope). Rejected otherwise at config validation.
    # STABILITY: AB2-eps Coriolis is conditionally stable in f·dt_mom — empirical
    # divergence threshold ≈0.55 (review probe). The ACC recipe grid spans
    # |lat|max≈44° ⇒ |f|max≈9.95e-5, f·dt_mom≈0.48 at dt_mom=4800 s — safely
    # inside, bounded at all probed friction levels. Configurations poleward of
    # ~55° at this dt_mom would exceed the threshold; check_coriolis_stability
    # warns at 0.5 and 0.55. Default
    # "matsuno_split" ⇒ BIT-IDENTICAL for every existing config.
    coriolis_scheme: str = "matsuno_split"
    # Energy-conserving (Sadourny vertex-f) C-grid Coriolis.  DEFAULT False keeps
    # the existing face-f stencil (f_u·v→u, f_v·u→v) BYTE-IDENTICAL.  When True,
    # both the barotropic-solver Coriolis and the Matsuno baroclinic sub-step use
    # the single shared VERTEX f so Σu·cor_u+Σv·cor_v == 0 on a β-plane (the
    # face-f form leaks ~1e-6·f·KE because f_u != f_v when f varies with lat —
    # the MITgcm barotropic-gyre oracle residual; see
    # docs/ocean/fidelity/mitgcm_gyre_energy_conservation.md).  On an f-plane the
    # two forms agree.  Applies to matsuno_split + implicit_cn/rigid_lid.
    coriolis_energy_conserving: bool = False

    # --- AB2 extrapolation scope (Veros-faithful dissipative placement) ---
    # Selects WHICH explicit tendencies the AB2 outer integrator extrapolates:
    #   "total" (DEFAULT, BIT-IDENTICAL) — legoESM's existing scheme: the AB2
    #     extrapolates the FULL explicit forward-Euler increment, INCLUDING the
    #     dissipative tendencies (momentum lateral friction + bottom drag;
    #     tracer lateral diffusion + GM/Redi isoneutral+skew diffusion).
    #   "advective" (VEROS-FAITHFUL) — only the ADVECTIVE part of the increment
    #     is AB2-extrapolated; the DISSIPATIVE tendencies are applied at WEIGHT
    #     1.0 (forward-Euler), matching Veros::
    #       X^{n+1} = X^n + (1.5+ε)·ΔX_adv^n − (0.5+ε)·ΔX_adv^{n-1} + 1.0·ΔX_diss^n
    #     Veros AB2-extrapolates ONLY {Coriolis, metric, advection, wind,
    #     p_hydro} for momentum (``core/external/solve_stream.py``) and ONLY
    #     advection for tracers (``vs.dtemp`` = ``advect_temperature``,
    #     ``core/thermodynamics.py``); the dissipative terms ride at weight 1.0:
    #     momentum lateral friction + bottom drag (added unextrapolated in
    #     ``solve_stream.py``) and tracer lateral+isoneutral+skew diffusion
    #     (added to ``tr[taup1]`` at weight 1.0 in ``thermodynamics.py`` /
    #     ``isoneutral/diffusion.py`` — all computed from the PRE-STEP tracer
    #     ``tr[tau]``).  In "advective" mode the AB2 carries
    #     ``{T,S,u,v}_incr_prev`` hold ONLY the advective increment (the carry
    #     semantics change is GATED — under "total" the carries are unchanged
    #     bit-identically).
    # WHY THIS MATTERS: the prior "total" scheme incurs a transient-only
    # O((0.5+ε)·Δstep(D)) error per dissipative tendency (it converges to the
    # SAME fixed point as "advective" — both reduce to a forward-Euler steady
    # balance) AND halves the AB2 stability margin on the NEGATIVE REAL AXIS
    # where stiff dissipation lives. STABILITY: at weight 1.0 the dissipative
    # update obeys the forward-EULER stability bound (|1 − λ·dt| ≤ 1, i.e.
    # λ·dt ≤ 2 for the real-negative eigenvalues of a diffusion operator); AB2
    # on the negative real axis is stable only to |λ·dt| ≲ 1 — so "advective"
    # is a STABILITY IMPROVEMENT for stiff dissipation, not a relaxation.
    # Composes cleanly with ``momentum_friction_additive`` (the implicit
    # VERTICAL friction is already weight-1.0 there): both on ⇒ the full Veros
    # dissipative scope (lateral + vertical friction + bottom drag all weight
    # 1.0).  Requires ``outer_integrator="ab2"`` (the scope is meaningless
    # without AB2; rejected otherwise at config validation).  Default "total"
    # ⇒ BIT-IDENTICAL for every existing config.
    ab2_scope: str = "total"
    # --- East-west cyclic-overlap projection (ORCA tripole seam) -----------
    # When True, the two longitude HALO columns are slaved to their ORCA
    # 2-point cyclic-overlap partners at the END of each step (cell-centred
    # fields: ``col[0] <- col[nx-2]``, ``col[nx-1] <- col[1]``), reconnecting
    # the east-west seam that the regular-grid roll-periodicity (period-nx)
    # leaves severed on an ORCA grid (physically period nx-2, with 2 overlap
    # halos).  The eORCA1 mesh marks the halo columns LAND, so the seam
    # (lon ~72.5E) carries a spurious wall; this + the matching mask/bathy/IC
    # overlap-fill at construction reconnects it.  ORCA-OVERLAP-SPECIFIC: only
    # valid on a grid whose first/last columns DUPLICATE columns nx-2 / 1 (the
    # tripole) -- WRONG on a genuinely period-nx regular lat-lon grid.  Default
    # False -> bit-exact for every existing grid/config; set only for tripole.
    ew_cyclic_overlap: bool = False
    # --- River-runoff depth spreading (NEMO rn_dep_max) --------------------
    # When > 0, the RUNOFF component of the freshwater forcing dilutes the
    # top ``runoff_depth_spread_m`` metres of the column (NEMO sbcrnf spreads
    # rivers over the top 150 m) instead of a single surface cell -- large
    # rivers (Amazon) otherwise sit too fresh/too shallow/too local.  The
    # column-integral salt tendency is unchanged (conservation identical);
    # only the vertical distribution moves.  0 = legacy top-cell (bit-exact).
    runoff_depth_spread_m: float = 0.0
    # --- Implicit (weight-1.0 pre-vmix) sponge placement (EXT-N2) -----------
    # Apply the SPONGE tracer relaxation (``SpongeForcing`` gamma·(ref − q))
    # at weight 1.0 inside the backward-Euler vertical-mixing solve instead of
    # summing it into the explicit (AB2-extrapolated) ``dT_dt``/``dS_dt`` —
    # matching Veros, which applies ``tempsalt_sources`` forward-Euler at
    # taup1 AFTER the AB2'd advection/diffusion and BEFORE the implicit vmix
    # (``veros/core/thermodynamics.py:419`` → ``veros/core/diffusion.py:
    # 132-141``: ``temp[taup1] += dt_tracer·temp_source·maskT``, NOT AB2'd).
    # The Rung-6 oracle tendency-match showed this placement CLASS
    # (weight-1-in-the-implicit-seam vs AB2'd-explicit) is a leading-order
    # fidelity term — the analogous implicit surface-forcing option dropped
    # the realized-T L2 mismatch 3.85 → 1.69.  The withheld rates ride
    # ``LatLonCGridOceanTendencies.tracer_source`` (NOT
    # ``surface_tracer_forcing``: the TKE surface buoyancy-flux
    # reconstruction must exclude column sources, as Veros keeps
    # tempsalt_sources out of forc_rho_surface).  The MOMENTUM sponge
    # (u_ref/v_ref) stays explicit (Veros has no momentum sponge).  Requires
    # ``implicit_vertical_mixing=True`` (rejected otherwise at config
    # validation).  Default False ⇒ the explicit stage-10c placement ⇒
    # BIT-IDENTICAL.
    sponge_forcing_implicit: bool = False
    # Lateral-friction CLOSURE selector (independent of the A_h/B_h/C_smag/C_leith
    # knobs above). "none" (default) → those knobs apply as usual. "om4p25" → the
    # GFDL OM4p25 Laplacian+biharmonic max(Smag,static) closure (Silvestri "SM2");
    # set the A_h/B_h/C_smag/C_leith knobs to 0 in that recipe so OM4p25 is the
    # sole lateral friction. Coefficients live in ``omp25`` (OMp25Config).
    lateral_friction_scheme: str = "none"
    omp25: object = None   # OMp25Config or None (defaults to OMp25Config() when scheme="om4p25")
    qg_leith_coeff: float = 2.0   # QG-Leith coefficient C (paper QG2 uses C=2); used when
    #   lateral_friction_scheme="qg_leith". HARMONIC ν=(C·Δ/π)³·√(|∇Q|²+|∇δ|²).
    # FULL QG2 (B5b): when True the baroclinic stretching term ∂_z(f/N²∇b) is added to the PV
    # gradient (∇q₁) with the Bachman grid-Burger/grid-Rossby min-bound — the faithful paper QG2.
    # Default False = BAROTROPIC ∇(ζ+f) (label "QG-Leith (barotropic)" in a comparison matrix).
    qg_leith_stretching: bool = False
    qg_leith_deformation_radius_m: float = 6.75e3   # L_d for the grid-Burger bound [m].
    # --- Veros u_centered dzw slot for the implicit vertical-diffusion solves ---
    # Selects the GRADIENT divisor (the center-to-center spacing) used by the
    # backward-Euler tracer (T/S) and momentum-friction vertical-diffusion solves:
    #   False (DEFAULT, BIT-IDENTICAL) — the midpoint reconstruction
    #     ``build_dz_half(dz_cell) = 0.5(dz_k + dz_{k+1})``.
    #   True (VEROS-FAITHFUL) — the coordinate's center-to-center spacing
    #     ``z_coord.dz_half_ref · J`` (Jacobian-scaled like every other
    #     thickness), i.e. Veros's ``dzw`` (thermodynamics.py:267
    #     ``delta = dt·kappaH/dzw``; same divisor for friction).  On a u_centered
    #     z-coordinate (the Veros-faithful ACC recipe) dz_half_ref alternates
    #     around the midpoint value exactly as Veros's dzw does (face ratios up to
    #     2.0 at the top face, ±10% below), so at IDENTICAL diffusivity the
    #     discrete flux differs per level.  This is the missed twin of the B3 slot
    #     fixes (N²/TKE/GM were moved to dz_half_ref·J; the implicit solves were
    #     not).  On a midpoint z-star coordinate dz_half_ref == build_dz_half(dz_ref)
    #     so the flag is a NO-OP there.  The CONTROL volume (dz_cell / dz_u / dz_v)
    #     is unchanged — only the gradient slot moves.  Requires
    #     ``implicit_vertical_mixing=True`` (rejected otherwise at config
    #     validation).  Default False ⇒ BIT-IDENTICAL for every existing config.
    implicit_vmix_dzw_slot: bool = False
    # --- Meridionally-FLAT (Oceananigans `Flat`-y topology) ---
    # When True, every meridional DIFFERENCE operator returns 0 — the faithful
    # legoESM analog of an Oceananigans `topology=(…, Flat, …)` dimension
    # (`δyᵃᶜᵃ(grid::Flat)=zero`).  v stays prognostic (so the Coriolis f×u→v
    # rotation works) but no ∂/∂y ever exists, so (a) there is NO meridional
    # pressure gradient → no closed-basin geostrophic locking of the barotropic
    # mode, and (b) NO 2Δy mode can form.  This is the faithful 2-D x–z setting the
    # Oceananigans internal_tide oracle uses; required to reproduce it (#576).
    # ``LatLonCGridOceanModel`` pushes this to the process-global
    # ``halo_latlon.set_meridionally_flat`` (the grid-operators backend flag, same
    # pattern as the halo backend) at construction.  Default False ⇒ BIT-IDENTICAL.
    meridionally_flat: bool = False
    # --- Astronomical (equilibrium) tidal forcing (OPT-IN barotropic body force) ---
    # Nested opt-in config (like `physics`/`gm_redi`): default-disabled instance =>
    # BIT-IDENTICAL. Consumed by ocean.physics.tidal_forcing.apply_tidal_forcing in
    # the barotropic momentum step (see that module's wiring note). Appended at the
    # NamedTuple tail so positional construction for legacy callers is preserved.
    tidal_forcing: TidalForcingConfig = TidalForcingConfig()

    @classmethod
    def from_flat(cls, **flat) -> "LatLonCGridOceanConfig":
        """Construct from FLAT keyword args (the legacy / YAML field names),
        distributing #501-grouped fields into their nested sub-configs.

        The canonical nested constructor is the NamedTuple itself
        (``LatLonCGridOceanConfig.from_flat(bottom_drag=DynBottomDragConfig(...), ...)``);
        this is the back-compatible flat entry point that existing call sites and
        the YAML loader use, so a caller can keep passing the flat
        ``bottom_drag_r=...`` and it is routed into ``bottom_drag``.  An unknown
        field raises (NamedTuple validates the residual kwargs) — typos stay
        loud.  Passing a sub-config object directly (``bottom_drag=...``) is
        also accepted (it falls through unchanged).
        """
        nested = {}
        _bd = {k: flat.pop(k) for k in DynBottomDragConfig._fields if k in flat}
        if _bd:
            nested["bottom_drag"] = DynBottomDragConfig(**_bd)
        _bt = {k: flat.pop(k) for k in BarotropicConfig._fields if k in flat}
        if _bt:
            nested["barotropic"] = BarotropicConfig(**_bt)
        _rc = {k: flat.pop(k) for k in RuntimeChecksConfig._fields if k in flat}
        if _rc:
            nested["runtime_checks"] = RuntimeChecksConfig(**_rc)
        _lv = {k: flat.pop(k) for k in LateralViscosityConfig._fields if k in flat}
        if _lv:
            nested["lateral_viscosity"] = LateralViscosityConfig(**_lv)
        _pf = {k: flat.pop(k) for k in PolarFilterConfig._fields if k in flat}
        if _pf:
            nested["polar_filter"] = PolarFilterConfig(**_pf)
        return cls(**nested, **flat)

    @classmethod
    def flat_fields(cls) -> frozenset:
        """The accepted FLAT field names (#501): the top-level fields with each
        grouped sub-config replaced by its member field names.  Drives both
        :meth:`from_flat` distribution and the flat ``ocean.*`` YAML typo-check,
        so the flat construction / YAML interface stays 1:1 with the pre-grouping
        field set even though the storage is nested.  Extend per nested group.
        """
        names = set(cls._fields) - {"bottom_drag", "barotropic", "runtime_checks",
                                    "lateral_viscosity", "polar_filter"}
        names |= set(DynBottomDragConfig._fields)
        names |= set(BarotropicConfig._fields)
        names |= set(RuntimeChecksConfig._fields)
        names |= set(LateralViscosityConfig._fields)
        names |= set(PolarFilterConfig._fields)
        return frozenset(names)

    def flat_get(self, name: str):
        """Read a field by its FLAT name (#501) — the read-side inverse of
        :meth:`from_flat` / :meth:`flat_fields`.  Resolves the grouped members
        through their nested sub-config, so reflective code that iterates flat
        field names (recipe-card audits, parity loops) keeps working:
        ``cfg.flat_get("barotropic_solver")`` == ``cfg.barotropic.barotropic_solver``.
        """
        if name in DynBottomDragConfig._fields:
            return getattr(self.bottom_drag, name)
        if name in BarotropicConfig._fields:
            return getattr(self.barotropic, name)
        if name in RuntimeChecksConfig._fields:
            return getattr(self.runtime_checks, name)
        if name in LateralViscosityConfig._fields:
            return getattr(self.lateral_viscosity, name)
        if name in PolarFilterConfig._fields:
            return getattr(self.polar_filter, name)
        return getattr(self, name)

    def replace_flat(self, **overrides) -> "LatLonCGridOceanConfig":
        """``_replace`` by FLAT field names (#501) — the ``_replace`` analog of
        :meth:`from_flat`.  Distributes grouped overrides into their nested
        sub-configs, so dict-splat override sites (recipe scheme presets, CLI
        overrides) that pass flat names keep working:
        ``cfg.replace_flat(A_h=0.0, barotropic_solver="rigid_lid")``.
        """
        nested = {}
        for sub_name, sub_cls in (("bottom_drag", DynBottomDragConfig),
                                  ("barotropic", BarotropicConfig),
                                  ("runtime_checks", RuntimeChecksConfig),
                                  ("lateral_viscosity", LateralViscosityConfig),
                                  ("polar_filter", PolarFilterConfig)):
            members = {k: overrides.pop(k) for k in list(overrides)
                       if k in sub_cls._fields}
            if members:
                nested[sub_name] = getattr(self, sub_name)._replace(**members)
        return self._replace(**nested, **overrides)
