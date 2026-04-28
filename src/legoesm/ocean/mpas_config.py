"""Configuration for MPAS Voronoi-mesh ocean model.

Provides config NamedTuples for the full 3D MPAS ocean dynamics
and for simplified (slab/fixed) ocean modes on Voronoi meshes.
"""

from __future__ import annotations

from typing import NamedTuple


class MPASOceanConfig(NamedTuple):
    """Configuration for MPAS ocean primitive equation solver.

    Fields
    ------
    g : float
        Gravitational acceleration [m/s²].
    rho_0 : float
        Reference seawater density [kg/m³].
    A_h : float
        Horizontal (harmonic) viscosity [m²/s].
    B_h : float
        Horizontal biharmonic viscosity [m⁴/s].  Applied as a constant-
        coefficient ``B_h * del4(u)`` operator.
    C_smag : float
        Smagorinsky coefficient (dimensionless, typical 0.01-0.15).
        When > 0, enables flow-dependent biharmonic Smagorinsky viscosity
        ``-del2(A_smag * del2(u))`` where ``A_smag = (C_smag * Δ)² |D|``.
    bottom_drag_r : float
        Linear bottom drag coefficient [m/s].  Applied as ``-r * u / dz_bot``
        at the bottom level in the baroclinic tendency and as
        ``-r * U_bar / H`` in the barotropic substeps.  Matches MITgcm's
        ``bottomDragLinear`` convention.
    K_h : float
        Horizontal tracer diffusivity [m²/s].
    K_bih : float
        Horizontal biharmonic tracer diffusivity [m⁴/s].  Applied as
        ``-K_bih * ∇⁴(tr)`` on cell-centered tracers using the scalar
        bilaplacian ``bilaplacian_cell_3d`` (two applications of the
        cell-centered scalar Laplacian ``div(grad(·))``).  Scale-selective
        dissipation with weaker damping of large scales than ``K_h``.
    A_v : float
        Vertical viscosity [m²/s].
    K_v : float
        Vertical tracer diffusivity [m²/s].
    n_barotropic_substeps : int
        Number of barotropic (free-surface) substeps per baroclinic step.
    pv_scheme : str
        PV flux scheme: "energy" or "enstrophy".
    use_conservation_fixer : bool
        Apply conservation fixers after each step.
    fix_volume : bool
        Fix volume (eta) conservation.
    fix_heat : bool
        Fix heat (T) conservation.
    fix_salt : bool
        Fix salt (S) conservation.
    min_water_column_m : float
        Minimum allowed water column depth [m].
    barotropic_damping : float
        Rayleigh damping coefficient for barotropic mode [1/s].
    freshwater_closure : str
        Freshwater closure: "none", "virtual_salt_flux", or "real_freshwater".
        "virtual_salt_flux": apply virtual salt flux to S, real mass flux to eta.
        "real_freshwater": apply real freshwater mass to eta and dilution to S.
        "none": ignore freshwater forcing.
    S_ref : float
        Reference salinity [PSU] for virtual salt flux.
    tracer_advection : str
        Tracer advection scheme: "upwind" or "tvd".
        "upwind" uses first-order donor-cell reconstruction.
        "tvd" uses second-order Van Leer limiter (less diffusive,
        monotone) for both horizontal and vertical advection.
    """
    g: float = 9.80616           # = constants.g
    rho_0: float = 1025.0        # = eos.rho_0
    A_h: float = 1.0e4
    B_h: float = 0.0
    C_smag: float = 0.0
    bottom_drag_r: float = 0.0
    K_h: float = 0.0
    K_bih: float = 0.0
    A_v: float = 1.0e-3
    K_v: float = 1.0e-4
    n_barotropic_substeps: int = 30
    # Default to enstrophy-conserving PV flux — avoids the ζ-checkerboard
    # null mode of the energy-conserving scheme (Ringler et al. 2010).
    # Use "energy" if total-KE conservation is required and the ζ null
    # mode can be controlled by other means.
    pv_scheme: str = "enstrophy"
    apvm_dt: float = 0.0  # APVM damping timescale [s]; set to baroclinic dt
                          # to enable the Anticipated PV Method upstream
                          # bias (damps ζ-checkerboard null mode of the
                          # energy-conserving PV flux). 0 = disabled.
                          # Set automatically by the test matrix to dt.
    pv_alpha: float = 1.0  # Weight on energy-conserving flux when
                           # ``pv_scheme == "mixed"``: α·energy + (1−α)·enstrophy.
                           # α=1.0 recovers pure energy; α=0.0 pure enstrophy.
                           # Ignored for pv_scheme in {"energy", "enstrophy"}.
    K_zeta_bih: float = 0.0  # Biharmonic dissipation on relative vorticity ζ
                             # [m⁴/s]. Adds −K_ζ·∇⁴ζ to the vorticity
                             # equation (scale-selective damping of grid-scale
                             # ζ patterns), applied as a tangential-gradient
                             # force on the momentum equation. Targets the
                             # ζ-checkerboard null mode of the energy-
                             # conserving PV flux — invisible to B_h·∇⁴u
                             # because the null mode lives in the kernel of
                             # the discrete curl. 0 = disabled.
    use_conservation_fixer: bool = False
    fix_volume: bool = True
    fix_heat: bool = True
    fix_salt: bool = True
    min_water_column_m: float = 0.5
    barotropic_damping: float = 0.0
    barotropic_diffusion_alpha: float = 0.01
    barotropic_diffusion_dt_ref: float = 60.0
    barotropic_div_damp: float = 0.0  # Divergence damping on barotropic velocity (dimensionless)
    bebt: float = 0.2               # Semi-implicit barotropic PGF [0,1]. 0=forward-backward, 0.2=MOM6 default.
    maxvel_barotropic: float = 0.0  # Velocity clipping [m/s]. 0=disabled.
    barotropic_time_filter: str = "cosine"  # "box" or "cosine"
    semi_implicit_coriolis: bool = True
    freshwater_closure: str = "virtual_salt_flux"
    S_ref: float = 35.0
    physics: object = None  # OceanPhysicsConfig or None
    eos: str = "wright"    # "wright" or "linear"
    eos_linear: object = None  # LinearEOSConfig when eos="linear"
    tracer_advection: str = "upwind"
    # Runtime bounds checks (matching cubed-sphere ocean)
    enable_runtime_checks: bool = False
    temperature_min_c: float = -5.0
    temperature_max_c: float = 45.0
    salinity_min_psu: float = 0.0
    salinity_max_psu: float = 50.0
    # Leith viscosity coefficient (Leith 1996).  When > 0 enables
    # flow-adaptive biharmonic viscosity ``-∇²(A_L ∇²u)`` with
    # ``A_L = (C_L · Δ_e)³ · |∇ζ|`` (or ``sqrt(|∇ζ|² + |∇δ|²)`` when
    # ``C_leith_modified = True``).  Typical values: 1.0–2.0.  Historically
    # appended at the END of the NamedTuple for positional-call stability.
    # Note: all in-tree constructors use keyword arguments and the
    # ``K_bih`` field inserted above deliberately mirrors the field order
    # of :class:`LatLonCGridOceanConfig`; positional stability is not
    # relied on at MPAS call sites.
    C_leith: float = 0.0
    C_leith_modified: bool = False


class MPASSimpleOceanConfig(NamedTuple):
    """Configuration for simplified ocean on Voronoi mesh.

    Mirrors SimpleOceanConfig but operates on (nCells,) arrays.

    Fields
    ------
    mode : str
        Ocean mode: "fixed", "slab", or "two_layer".
    sst_constant : float
        Fixed SST [K] (for "fixed" mode).
    h_mix : float
        Mixed layer depth [m].
    rho_ocean : float
        Seawater density [kg/m³].
    c_ocean : float
        Specific heat of seawater [J/(kg·K)].
    Q_flux : float
        Prescribed ocean heat flux [W/m²].
    albedo_ocean : float
        Ocean albedo.
    emissivity_ocean : float
        Ocean emissivity.
    Cd_ocean : float
        Drag coefficient over ocean.
    Ch_ocean : float
        Heat transfer coefficient over ocean.
    U_min : float
        Minimum wind speed [m/s].
    T_freeze : float
        Freezing temperature [K].
    h_deep : float
        Deep layer depth [m] (for "two_layer" mode).
    k_mix : float
        Vertical mixing coefficient [m²/s] (for "two_layer" mode).
    restore_deep : bool
        Restore deep layer toward reference (for "two_layer" mode).
    T_deep_ref : float
        Deep layer reference temperature [K].
    tau_deep : float
        Deep layer restoring timescale [s].
    """
    mode: str = "fixed"
    sst_constant: float = 300.0
    h_mix: float = 50.0
    rho_ocean: float = 1025.0
    c_ocean: float = 3994.0
    Q_flux: float = 0.0
    albedo_ocean: float = 0.06
    emissivity_ocean: float = 0.97
    Cd_ocean: float = 1.5e-3
    Ch_ocean: float = 1.5e-3
    U_min: float = 1.0
    T_freeze: float = 271.35
    h_deep: float = 200.0
    k_mix: float = 1.0e-4
    restore_deep: bool = False
    T_deep_ref: float = 278.0
    tau_deep: float = 365.25 * 86400.0
