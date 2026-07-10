"""Configuration for MPAS Voronoi-mesh ocean model.

Provides config NamedTuples for the full 3D MPAS ocean dynamics
and for simplified (slab/fixed) ocean modes on Voronoi meshes.
"""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants
from legoesm.ocean.eos import FreezingPointConfig


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
    bottom_drag_bbl_thickness : float
        BBL (bottom boundary layer) thickness [m] for distributed-drag
        formulation (Killworth & Edwards 1999 / MOM6 ``BBL_thick_min``).
        When > 0, the bottom drag stress is spread over the band
        ``[z_seafloor, z_seafloor + H_BBL]`` instead of being applied
        to a single (possibly very thin) partial cell.  At the thinnest
        partial cells (~0.3 m on ETOPO+ico4), the legacy single-cell
        drag CFL-violates explicitly; the BBL formulation bounds the
        per-cell tendency by ``r·u/H_BBL``.  Default ``0.0`` keeps the
        legacy single-cell drag for back-compat; **for realistic
        bathymetry use 50.0 m or so** (matches the lat-lon convention).
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
    g: float = constants.g
    rho_0: float = constants.rho_ocean
    A_h: float = 1.0e4
    B_h: float = 0.0
    C_smag: float = 0.0
    C_smag_lap: float = 0.0  # Laplacian Smagorinsky coefficient (dimensionless).
                              # When > 0, adds flow-dependent Laplacian viscosity
                              # A_smag = (C_smag_lap · Δ)² · |D| where |D| is
                              # the strain-rate magnitude.  Effective at coarse
                              # resolution (~120 km) where the biharmonic
                              # Smagorinsky (C_smag) produces zero.  Typical
                              # values: 0.1–0.3.  Matches the lat-lon
                              # ``C_smag_lap`` scheme.
    bottom_drag_r: float = 0.0
    bottom_drag_bbl_thickness: float = 0.0
    bottom_drag_bg_velocity: float = 0.0  # Background velocity floor for
                                           # quadratic drag [m/s].  When > 0,
                                           # upgrades linear drag to quadratic-
                                           # with-floor (MOM6 DRAG_BG_VEL):
                                           #   r_eff = (r / u_bg) * sqrt(u² + u_bg²)
                                           # Recovers linear r at |u| → 0,
                                           # quadratic Cd*|u| at |u| >> u_bg
                                           # where Cd = r / u_bg.
                                           # Typical: u_bg = 0.1 m/s with
                                           # r = 1e-3 gives Cd = 0.01.
                                           # 0 = legacy linear drag (bit-exact).
    # NEMO zdfdrg drag laws (mirror of the lat-lon DynBottomDragConfig):
    # "legacy" keeps the r/bg_velocity path bit-exact; "nemo_quadratic" /
    # "nemo_loglayer" compute r = Cd·√(2·KE_bot + ke0) from the bottom-cell
    # Ringler KE averaged to edges (loglayer: Cd from the edge bottom
    # thickness, clip((κ/ln(½h/z0))², cd0, cdmax)).
    bottom_drag_scheme: str = "legacy"
    bottom_drag_cd0: float = 1.0e-3     # NEMO rn_Cd0 [-]
    bottom_drag_cdmax: float = 0.1      # NEMO rn_Cdmax [-]
    bottom_drag_z0: float = 3.0e-3      # NEMO rn_z0 [m]
    bottom_drag_ke0: float = 2.5e-3     # NEMO rn_ke0 [m²/s²]
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
    barotropic_u_viscosity: float = 0.0  # Lateral viscosity on u_bar [m²/s].
                                         # Damps the TRiSK rotational null branch
                                         # (Thuburn 2008; Ringler et al. 2010) on
                                         # hexagonal C-grids — invisible to eta
                                         # diffusion and divergence damping.
                                         # 0 = disabled; typical 1e3-1e4 m²/s
                                         # for global ico4 (~460 km) meshes.
                                         # PARTIAL-CELL ETOPO: prefer
                                         # barotropic_u_biharmonic instead —
                                         # del2 at the magnitudes needed to
                                         # damp the partial-cell rotational
                                         # null mode (~3e6) over-damps real
                                         # mesoscale flow.
    equatorial_visc_boost: float = 0.0  # Multiplicative boost on
                                         # lateral viscosity at low
                                         # latitudes, applied per-edge as
                                         # ``A_eff = A · (1 + boost ·
                                         # cos²(lat_edge))``.  Affects
                                         # BOTH ``A_h`` (3D momentum) and
                                         # ``barotropic_u_viscosity``
                                         # (depth-mean).  Targets the
                                         # equatorial f→0 mode that the
                                         # implicit-CN solver's Coriolis
                                         # predictor-corrector cannot
                                         # damp (project_mpas_etopo_
                                         # instability.md §"equatorial
                                         # mode").  Mirrors the lat-lon
                                         # ``A_h_lat_scaling`` mechanism
                                         # but with controllable strength.
                                         # Typical 3-10 for ico4 ETOPO.
                                         # 0 = uniform viscosity.
    use_h_actual_pgf: bool = True          # Integrate baroclinic pressure
                                            # cumsum against actual partial-
                                            # cell thickness h_k rather
                                            # than the reference dz_ref.
                                            # Matches NEMO ``ln_hpg_zps``,
                                            # MITgcm, and our own lat-lon
                                            # implementation.  Required for
                                            # Adcroft PGF consistency: the
                                            # AC correction computes centroid
                                            # depths from h_partial, so
                                            # p_prime must also be on the
                                            # h_partial grid.  When False,
                                            # p_prime uses dz_ref while AC
                                            # uses h_partial → mismatch →
                                            # spurious PGF at step edges.
                                            # SCHEME COUPLING: this flag
                                            # takes effect ONLY for
                                            # ``pgf_scheme="adcroft"`` (its
                                            # matched partner).  The bare
                                            # "centered" scheme has no face
                                            # correction, so h_actual p_prime
                                            # would difference pressures at
                                            # mismatched centroid depths
                                            # across a bottom-level step and
                                            # go unstable; centered always
                                            # uses the dz_ref reference-depth
                                            # grid regardless of this flag.
                                            # smc03/ahh08/zero drop p_prime,
                                            # so it is moot for them.
    use_baroclinic_rho_ref: bool = False  # DYNAMIC (legacy / discouraged):
                                           # subtracts ρ_ref(z) computed
                                           # as wet-cell mean of ρ on
                                           # EVERY tendency call.  NaN'd
                                           # at day 60 on ETOPO+ico4
                                           # because ρ_ref chases T,S
                                           # drift (positive feedback).
                                           # Use ``use_static_baroclinic
                                           # _rho_ref`` instead.  Kept
                                           # only for back-compat /
                                           # diagnostic comparison.
    use_static_baroclinic_rho_ref: bool = False
        # STATIC (preferred):
        # at init, compute ρ_ref(z) from EOS(T_init, S_init) averaged
        # over wet cells per level, FREEZE it on ``state.rho_ref_z``,
        # and use it as ``ρ' = ρ − ρ_ref(z)`` in every PGF call.  Cuts
        # the partial-cell PGF residual (the seed of the bottom-trapped
        # rotational mode) ~24× without the dynamic version's drift
        # feedback (project_mpas_etopo_instability.md §8g — Option B).
        # Off by default (back-compat: ρ' = ρ − ρ_0).  Mutually
        # exclusive with ``use_baroclinic_rho_ref``.
    barotropic_u_biharmonic: float = 0.0  # Biharmonic ∇⁴ damping on u_bar
                                          # [m⁴/s].  Scale-selective: damps
                                          # grid-scale modes much harder than
                                          # mesoscale, so safe to use at
                                          # production strength on partial-
                                          # cell topography.  Wired in the
                                          # implicit-CN path only.  Typical
                                          # 1e15-1e17 m⁴/s for ico4.
    bebt: float = 0.2               # Semi-implicit barotropic PGF [0,1]. 0=forward-backward, 0.2=MOM6 default.
    maxvel_barotropic: float = 0.0  # Velocity clipping [m/s]. 0=disabled.
    barotropic_time_filter: str = "cosine"  # "box" or "cosine"
    semi_implicit_coriolis: bool = True
    # Barotropic solver selection.  ``"explicit_substep"`` (default) uses
    # the existing forward-backward substep loop with cosine filter.
    # ``"implicit_cn"`` uses a single-step Crank-Nicolson free surface
    # with PCG Helmholtz solve, mirroring the lat-lon implementation
    # (docs/dev-notes/issues/barotropic_mode_noise.md).  Eliminates the TRiSK
    # rotational null branch (Thuburn 2008; Ringler+ 2010 §6) by
    # construction; no substepping or time filter needed.
    barotropic_solver: str = "explicit_substep"
    # Implicit-CN knobs (only used when ``barotropic_solver = 'implicit_cn'``).
    # 0.5 = pure Crank-Nicolson (2nd-order, no implicit damping); 1.0 =
    # fully backward (1st-order, max damping).  0.55 is the standard
    # MITgcm/MPAS-O choice — slightly past CN to suppress chequerboard
    # while staying close to 2nd-order in time.
    barotropic_implicit_theta_eta: float = 0.55
    barotropic_implicit_theta_pgf: float = 0.55
    barotropic_implicit_pcg_tol: float = 1.0e-10
    barotropic_implicit_pcg_maxiter: int = 200
    # Distributed (MPI) implicit-CN knobs, mirroring the lat-lon
    # ``LatLonCGridOceanConfig`` contract.  The distributed Voronoi PCG IS
    # wired (barotropic_implicit_mpas.py dispatches at entry when
    # ``initialize_voronoi_mpi`` armed a partition layout): a cell-halo
    # exchange composed into every ``A_op`` + owned-cell-masked
    # area-weighted dots (docs/ocean_experiments/distributed_barotropic_pcg
    # .md).  ``fixed_iters`` targets 1e-10 residual on the diagonally-
    # dominant Voronoi Helmholtz.  ``pcg_variant`` selects the reduction
    # strategy ("standard" 2-dot PCG, or "single_reduce" Chronopoulos–Gear
    # with one batched allreduce per iteration — validated at solver entry,
    # ValueError on unknown).
    barotropic_implicit_pcg_fixed_iters: int = 60
    barotropic_implicit_pcg_residual_tol: float = 1.0e-10
    barotropic_implicit_pcg_variant: str = "standard"
    freshwater_closure: str = "virtual_salt_flux"
    normalize_freshwater: bool = False  # When True, subtract the global
                                        # area-weighted mean freshwater flux
                                        # at each step so that net deta/dt
                                        # integrates to zero globally.
                                        # Standard OMIP practice for runs
                                        # without a sea-ice model where the
                                        # P-E+R budget doesn't close.
                                        # Preserves the spatial pattern of
                                        # forcing; only removes the global
                                        # imbalance.
    S_ref: float = 35.0
    physics: object = None  # OceanPhysicsConfig or None
    eos: str = "wright"    # "wright" or "linear"
    eos_linear: object = None  # LinearEOSConfig when eos="linear"
    gm_redi: object = None  # GMRediConfig — None disables GM/Redi.  Only the
                            # 'centered' slope_scheme is implemented on MPAS;
                            # 'triads' raises NotImplementedError (Phase 5
                            # of docs/ocean/experiments/gm_redi_mpas_plan.md).
    mle: object = None     # MLEConfig — None disables Fox-Kemper mixed-layer-eddy
                           # restratification.  Voronoi bolus port of NEMO nn_mle=1
                           # (docs/ocean/experiments/mle_mpas_port_plan.md).
    implicit_vertical_mixing: bool = True  # When True, vertical viscosity
                                           # (A_v on velocity) and vertical
                                           # diffusivity (K_v on tracers) are
                                           # applied via backward-Euler implicit
                                           # solve in step(), NOT as explicit
                                           # tendencies. Unconditionally stable
                                           # — required for realistic A_v at
                                           # coarse vertical resolution (#204).
                                           # When False, reverts to the legacy
                                           # explicit vertical diffusion in the
                                           # tendency function.
    tracer_advection: str = "upwind"  # "upwind", "tvd" (Van Leer), "superbee" (Sweby/Veros)
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
    # Pressure-gradient scheme (P3 of MPAS realistic-geometry plan;
    # see ``docs/ocean/experiments/realistic_geometry_mpas_plan.md``).
    # ``"centered"`` (default) is the legacy bare-gradient
    # ``gradient_edge(p'/rho_0)`` — correct for flat-bottom z-star but
    # produces O(1 cm/s) spurious shelf-break currents on partial cells.
    # ``"adcroft"`` adds the Adcroft & Campin (2004) face correction
    # that shifts each cell's pressure to the shallower of the two
    # cell centroids before differencing — bit-exact zero on full
    # cells, eliminates the partial-cell PGF cancellation error.
    # CVT mesh required for ``"adcroft"`` (see helper docstring).
    # ``"smc03"`` enables the Shchepetkin & McWilliams (2003)
    # density-Jacobian PGF (per-column harmonic-slope ρ(z)
    # reconstruction evaluated at a face-reference depth) — required
    # for stability on real bathymetry; see
    # ``docs/ocean/experiments/density_jacobian_pgf_mpas.md``.  Only
    # meaningful with ``OceanPartialCellCoordinate``; falls back to
    # centered on z-star.
    # ``"ahh08"`` enables the Adcroft, Hallberg & Hill (2008) analytic
    # finite-volume PGF: closed-form ``∫p dz`` per cell using the
    # Wright EOS rational form, then face-averaged-pressure
    # differencing over the common wet face on each edge.  Gives
    # machine-zero rest-state PGF residual on partial cells regardless
    # of step structure (the property SMC03 only achieves on linear
    # ρ(z)).  Requires ``eos='wright'``.  Only meaningful with
    # ``OceanPartialCellCoordinate``.  See plan-doc §8i.
    # ``"zero"`` zeros both ``p'/rho_0`` and any partial-cell correction
    # — diagnostic only (per §8f / §8j PGF=0 tests).  Should never run
    # in production: removes the PGF entirely.
    # Allowed values: ``"centered" | "adcroft" | "smc03" | "ahh08" | "zero"``.
    pgf_scheme: str = "centered"
    # Threshold for the ``vertex_thickness_hybrid`` min-rule fallback in
    # the TRiSK PV term ``q = ζ/h_v``.  At each (vertex, level), if
    # ``min_wet_h < alpha * max_wet_h`` the function returns the min-
    # over-active cell thickness (MITgcm hFacZ convention); otherwise
    # the active-renormalized kite-area mean (Petersen 2015 / MPAS-O
    # production).  ``alpha=0.5`` (legacy) was added on a previous
    # internal audit's recommendation; the 2026-05-04 audit found the
    # min-rule branch *amplifies* ``q`` at deep partial-cell step
    # vertices by O(h_max/h_min) — exactly the topographic-step regime
    # where the bottom-trapped instability lives.  ``alpha=0.0``
    # disables the min branch entirely (always uses kite-mean) and is
    # the recommended setting for partial-cell ETOPO runs.
    # Only meaningful with ``OceanPartialCellCoordinate``.
    vertex_thickness_alpha: float = 0.5
    # Sea-ice thermodynamic surrogate: floor the SURFACE SST at the seawater
    # freezing point (~-1.8 C) when ``freeze_floor`` is set.  legoESM carries no
    # prognostic ice, so without it the Arctic over-cools ~4 C below NEMO; the
    # floor closes that gap (the same clamp LatLonCGridOceanConfig.freeze_floor
    # applies).  Off by default so conserving runs are bit-exact unaffected.
    freeze_floor: bool = False
    freeze_floor_temp_c: float = constants.T_freeze_ocean - constants.T_freeze
    # Seawater freezing-point (liquidus) scheme for the freeze_floor above.
    # "constant" (default) keeps freeze_floor_temp_c byte-identical; a liquidus
    # scheme makes the surface floor track the LOCAL surface salinity.  Consumed
    # by ocean_model_mpas._step_impl freeze block.  MED-1.
    freezing: FreezingPointConfig = FreezingPointConfig()
    # River-runoff depth spreading (NEMO sbcrnf rn_dep_max): when > 0 the
    # runoff freshwater dilutes the top this-many metres instead of a single
    # surface cell (Amazon plume fidelity).  Column-integral salt tendency
    # unchanged; 0 = legacy top-cell (bit-exact).
    runoff_depth_spread_m: float = 0.0


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
    freezing : FreezingPointConfig
        Seawater freezing-point (liquidus) scheme for the slab/two-layer
        freeze clamp AND its ``Q_freeze`` diagnostic.  ``"constant"``
        (default) keeps ``T_freeze`` above byte-identical; a liquidus scheme
        is evaluated at ``constants.S_ocean_ref`` (the slab carries no
        prognostic salinity) via the shared owner
        ``eos.slab_freeze_point_K`` — mirrors
        ``SimpleOceanConfig.freezing``.  MED-1 follow-up.
    """
    mode: str = "fixed"
    sst_constant: float = 300.0
    h_mix: float = 50.0
    rho_ocean: float = constants.rho_ocean
    c_ocean: float = constants.c_sw
    Q_flux: float = 0.0
    albedo_ocean: float = 0.06
    emissivity_ocean: float = 0.97
    Cd_ocean: float = 1.5e-3
    Ch_ocean: float = 1.5e-3
    U_min: float = 1.0
    T_freeze: float = constants.T_freeze_ocean
    h_deep: float = 200.0
    k_mix: float = 1.0e-4
    restore_deep: bool = False
    T_deep_ref: float = 278.0
    tau_deep: float = 365.25 * 86400.0
    # Appended at the END to preserve positional construction (same convention
    # as LatLonCGridOceanConfig field additions).
    freezing: FreezingPointConfig = FreezingPointConfig()
