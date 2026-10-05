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
        Tracer advection scheme: "upwind", "tvd" or "superbee". An unknown
        value raises.
        "upwind" uses first-order donor-cell reconstruction.
        "tvd" uses the second-order Van Leer limiter for both horizontal
        and vertical advection; "superbee" uses the more compressive Sweby
        limiter. Both are 1-D TVD limiters applied direction by direction,
        so neither is multi-dimensionally monotone: MEASURED on the
        resolved Petersen lock exchange (on the STRUCTURED arm, where the
        same schemes are available) they undershoot an initial [5, 30] degC
        range to -0.40 and -2.99 degC. There is NO flux-corrected (FCT)
        scheme in this port, which is what that case needs -- so a
        cross-grid lock exchange at resolving resolution cannot share an
        advection family with the lat-lon arm today.
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
    # eta-floor clamp redistribution refinements PER CALL (2 calls per
    # barotropic substep; each iteration costs one batched global
    # reduction, so the substep scan pays 2*n_substeps*iters messages
    # per step — the measured ocean-CPU rank-count-term driver class).
    # 3 = the historical exact-redistribution default; 1 = scaling
    # experiment knob (positivity unaffected — the clamp's final
    # maximum() holds regardless; the un-refined mass residual is
    # absorbed by the step-level conservation fixer).
    eta_floor_clamp_iters: int = 3
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
    K_zeta_bih: float | None = 0.0
                             # ``None`` = DERIVE from the mesh (see
                             # ``resolution_scaled_k_zeta_bih``); ``0.0`` = off;
                             # any other value pins the coefficient.
                             # Biharmonic dissipation on relative vorticity ζ
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
                                         # exp(-½(lat_edge/σ)²))`` with σ =
                                         # ``equatorial_visc_sigma_deg``.  Applied
                                         # to ``A_h`` (3D momentum) and to the
                                         # IMPLICIT-CN ``barotropic_u_viscosity``
                                         # (depth-mean); the explicit-substep
                                         # barotropic solver uses uniform
                                         # viscosity (no boost).  Targets the
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
    # strategy ("standard" 2-dot PCG, "single_reduce" Chronopoulos–Gear
    # with one batched allreduce per iteration, or "single_reduce_deep", the
    # same with the halo exchanged every few iterations — validated at solver
    # entry, ValueError on unknown).
    #
    # History (superseded 2026-10-05 by gpoly@20, see the end of this block):
    # 20 with the "poly" preconditioner below (owner decision 2026-09-20,
    # A/B at 32 and 128 GPUs: step -8%/-14.5% f32, -6%/-11% f64 against
    # Jacobi at 30, same residual).  The Jacobi history that set 30:
    # measured on the REAL captured systems (scripts/validate/
    # ocean_fidelity/barotropic_pcg_convergence.py) at subdivision 7, 8 and
    # 9, in both precisions, after 200 spin-up steps at dt = 300 s.  Over
    # the configurations tested, the mesh set the iteration count and the
    # column did not: subdivision 9 at 10, 20 and 40 levels gives the same
    # table, while subdivision 8 is at its floor ~10 iterations sooner than
    # 9.  Timestep, bathymetry and state were NOT varied and could move it.
    #
    # At subdivision 9 (2.6M cells, the production/scaling mesh), relative
    # residual and max|b - A eta|/dt.  That second column is the WORST LOCAL
    # continuity-defect rate a cell carries in one step, not a measured
    # global mass drift — the cell-to-cell and step-to-step cancellation is
    # unmeasured, so read it as an upper bound on how wrong one cell's
    # free-surface tendency can be, not as sea level lost per day:
    #     M      float32              float64
    #     20     1.0e-5  / 12 mm/day  3.0e-6  / 11 mm/day
    #     30     3.2e-6  / 1.4        1.0e-7  / 0.30
    #     40     3.3e-6  / 1.4        3.2e-9  / 0.011
    #     60     3.3e-6  / 1.4        3.5e-12 / 5e-6
    # float32 reaches its own precision floor at 30 and buys nothing after
    # it.  float64 keeps descending, so 30 trades sea-level budget accuracy
    # (0.3 mm/day per step, still 4x under the float32 floor the model
    # already runs at) for ~1.8x on step time.  Owner decision 2026-09-19;
    # raise to 40 if a long float64 integration shows mass drift.
    #
    # NOT covered by that measurement, and the two reasons to revisit this:
    # no multi-rank convergence check (the probe replays the captured system
    # on one process, where the halo exchange is the identity); and no
    # gradient comparison through the unrolled adjoint, which is the use
    # this most plausibly harms — reverse mode differentiates the TRUNCATED
    # algorithm exactly, so a loose forward residual bounds nothing about
    # the derivative, and a training run would degrade without any mass
    # diagnostic firing.  A verification run that must hit the configured
    # 1e-10 residual has to set the count back up explicitly.
    # The lat-lon C-grid default (state.py) is a different operator on a
    # different mesh and stays at 60 until measured.
    #
    # 20 with the GLOBAL polynomial ("gpoly", below) since 2026-10-05 (owner
    # decision). gpoly@15 was faster (-9/-12/-11% at 8/32/128 GPUs,
    # single_reduce) and beat poly@20 on L9 at dt=300 s, but FAILED the
    # finer-mesh check (job 27848794): on L10 @512 emulated devices and on
    # L9 at dt=600 s it is slightly worse than poly@20 (f64 rel_res 4.3e-4
    # vs 3.9e-4; 1.3e-3 vs 1.0e-3), while gpoly@20 is 4-6x better than
    # poly@20 there at about the same step cost (gpoly vs poly at 20:
    # +0.5..+1.5% at 8/32 GPUs, -1% at 128). NOTE: at L10 or dt=600 s every
    # option is far from converged at 20 (rel_res ~1e-4..1e-3); the count is
    # mesh/dt-specific. Single-device runs are unaffected (stock CG).
    # The opt-in deep-halo Jacobi solver (pcg_variant below) needs 30: Jacobi
    # reaches the 1e-10 relative residual in 27-30 iterations on real s7
    # systems (spin-up 30/150/400 steps, 128 emulated ranks; identical at 16).
    barotropic_implicit_pcg_fixed_iters: int = 20
    barotropic_implicit_pcg_residual_tol: float = 1.0e-10
    # "standard" (two allreduces per iteration) is the default since
    # 2026-10-02 (owner decision): it is the recurrence gpoly x 15 was
    # measured with (GPU weak ladder -6 / -8.5 / -10.4 % at 8 / 32 / 128
    # GPUs); gpoly x 15 with "single_reduce" has not been measured.
    # History: "single_reduce" (Chronopoulos-Gear, one batched allreduce per
    # iteration instead of two) was the default 2026-09-26..2026-10-02. It
    # had been owner-approved after two checks: convergence on the REAL captured systems is identical (s7 L40,
    # poly:4, 16-device local preconditioner: rel. residual 4.5e-16 at 20
    # iterations for both, |eta_single - eta_standard| 7e-18 m); and on
    # Derecho CPU with the butterfly global sum, s7 16 ranks/node, 313 vs 315
    # ms/step at 1 node, 167 vs 171 at 2, 110 vs 118 at 4, 86.9 vs 97.2 at 8.
    # float32 checked too (same systems): both recurrences reach the f32
    # residual floor 1.56e-7 by 15 iterations, eta differs by 3.7e-9 m.
    # GPU (NCCL) was not re-measured with it.
    # "single_reduce_deep" (opt-in): the same recurrence with the cell halo
    # of (r, s) exchanged once every `complete_cell_rings` iterations instead
    # of every iteration; the halo is recomputed redundantly.  Owned results
    # equal the per-iteration exchange bit for bit (x64 test).  Needs precond
    # "jacobi" and fixed_iters=30.  Cadence = rings the layout certifies: 3 on
    # the SPMD layout, 1 on the MPI-per-rank layout (its halo has no closure
    # passes, so there it is the per-iteration exchange of two vectors).
    # Derecho CPU s7 L40 f64, 3 repeats, ms/step: 8 nodes 82.8 (poly4 x 20)
    # -> 75.8 with deep Jacobi x 30; 2 nodes 164.2 -> 152.8; Jacobi x 30
    # with per-iteration exchange 92.7 / 165.3.  Not the default: owner
    # decision 2026-10-02 keeps gpoly x 15 (no head-to-head measurement).
    barotropic_implicit_pcg_variant: str = "standard"
    # Distributed-only preconditioner for the fixed-iteration PCG
    # (default "gpoly" since 2026-10-02, see below; the MPI Voronoi lane
    # must select "poly"). "jacobi" or "poly": a communication-free Neumann-series
    # polynomial in the device-local block of A (K local mat-vecs, no
    # halo exchange, so it costs nothing in ppermute rounds and buys
    # iterations back).  Measured on the real subdivision-9 systems,
    # 128 emulated devices, f64, relative residual:
    #                iters=10    15        20        30
    #     jacobi       7.9e-05   1.5e-05   3.1e-06   9.9e-08
    #     local poly4  2.0e-05   1.4e-06   8.9e-08   4.0e-10
    #     local poly8  9.1e-06   4.0e-07   1.6e-08   2.6e-11
    # Each PCG iteration still costs one cell-halo exchange plus two
    # allreduces; the win is reaching the target residual at a smaller
    # ``fixed_iters`` (30 -> 20 at poly4).
    # Default "poly" 2026-09-20..2026-10-02 (owner decision, A/B above);
    # "jacobi" is the pre-2026-09-20 solver and needs fixed_iters=30 for the
    # same residual (and is what "single_reduce_deep" requires).
    # Default "gpoly" since 2026-10-02 (owner decision; see fixed_iters for
    # the A/B): the same polynomial on the GLOBAL operator, evaluated
    # redundantly on the SPMD halo (layout halo_depth >= sweeps-2, which the
    # historical 2 satisfies at 4 sweeps); one exchange per iteration as
    # before, and the answer no longer depends on the device count. Same
    # systems, 128 emulated devices: f64 rel_res 2.3e-6 / 6.4e-8 / 1.8e-9 at
    # iters 10 / 15 / 20. SPMD lane only: the MPI Voronoi lane refuses it
    # (select "poly" there) and a single device keeps the stock CG solve.
    barotropic_implicit_pcg_precond: str = "gpoly"
    barotropic_implicit_pcg_poly_sweeps: int = 4
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
    # NEMO ln_rnf_depth_ini per-cell spread-depth MAP [m] (array shape
    # (nCells,) on the flattened Voronoi state, or None): depth proportional
    # to the local climatological runoff maximum so small Arctic/Siberian
    # rivers stay near-surface (more shelf-surface freshening) while the
    # Amazon spreads to ~150 m — a FLAT runoff_depth_spread_m dilutes small
    # river plumes through the whole shelf column and leaves the Siberian/
    # Beaufort SSS several PSU too salty.  Build with
    # forcing.runoff_depth.nemo_runoff_depth_map; mutually exclusive with a
    # nonzero runoff_depth_spread_m (resolve_runoff_spread_arg validates).
    # Column-integral salt tendency unchanged; None = legacy top-cell.
    runoff_depth_spread_map: object = None
    # Positional-stability tail: append new fields here (never mid-class).
    equatorial_visc_sigma_deg: float = 5.0  # Gaussian half-width [deg lat] of
    # the equatorial viscosity boost (see ``equatorial_visc_boost``); the
    # instability is confined to |lat| < ~10°, so the default 5° matches the
    # lat-lon production config. Only used when ``equatorial_visc_boost > 0``.
    # Salinity the virtual-salt closure multiplies the freshwater flux by:
    # "s_ref" (default, bit-identical) = the fixed scalar above; "local" =
    # the LOCAL top-cell salinity (NEMO tra_sbc: sfx = emp * sss) — removes
    # the fresh-shelf over-brining of the fixed-35 closure (2026-07-18
    # Arctic halocline-erosion audit).  Mirrors the lat-lon C-grid field.
    freshwater_salinity: str = "s_ref"   # "s_ref" | "local"

    # --- resolution scaling of the biharmonic vorticity damping -------------
    # Appended at the END of the schema on purpose: positional construction of
    # this NamedTuple stays valid for every existing caller.
    # Anchor for the DERIVED ``K_zeta_bih`` (``K_zeta_bih=None``): the value
    # and the mesh spacing it was tuned at.  ``1e14`` is the OMIP NEMO-match
    # MPAS recipe's coefficient, tuned on the ico6 mesh (SST RMSE 0.84 vs
    # NEMO).  The spacing is that mesh's mean ``dcEdge`` at the DEFAULT 50
    # Lloyd iterations, to full float precision, so an ico6 run reproduces
    # ``K_zeta_bih_ref`` EXACTLY rather than to 0.1 % (measured: the ico6 mesh
    # built by ``create_voronoi_mesh(6)`` derives 1.0e14 bit-identically).
    # The same mesh at Lloyd 0 measures 120324.32 m, which the cubed law turns
    # into 0.3 % on the coefficient.
    K_zeta_bih_ref: float = 1.0e14                 # [m⁴/s] at the reference spacing
    K_zeta_bih_ref_dx_m: float = 120194.60581296285  # [m] ico6 mean dcEdge, Lloyd 50
    # Provenance of a DERIVED coefficient, stamped by the model constructor:
    # the mesh spacing it was derived from (0.0 = pinned, not derived), so a
    # saved configuration records WHICH mesh its coefficient belongs to and a
    # replay on another mesh is visible rather than a silent re-pin.
    K_zeta_bih_dx_m: float = 0.0

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



def resolution_scaled_k_zeta_bih(dx_mean_m: float, config: "MPASOceanConfig") -> float:
    """Biharmonic vorticity damping coefficient for a mesh of spacing ``dx_mean_m``.

    ``K_ζ`` damps the ζ-checkerboard null mode of the energy-conserving PV
    flux.  Held fixed across resolutions it is BOTH over-dissipative and
    explicitly unstable on a finer mesh: the ∇⁴ operator's stability limit
    scales as ``Δ⁴/K_ζ``, so a coefficient tuned at ~120 km violates it at
    28 km (measured: an ico8 cold start diverges at step 11 with the fixed
    1e14 m⁴/s, and survives with the coefficient scaled).

    The scaling is ``K_ζ(Δ) = K_ref · (Δ/Δ_ref)³`` — the NEMO ``ldf_dyn``
    convention for a biharmonic operator, which holds the damping's velocity
    scale ``K_ζ/Δ³`` constant and lets the stability limit grow LINEARLY with
    refinement, rather than the ``Δ⁴`` law that would hold the limit fixed and
    let the damping vanish.  ``config.K_zeta_bih_ref`` / ``K_zeta_bih_ref_dx_m``
    carry the anchor, so the choice is in the configuration, not in a default
    buried at a call site.

    Returns ``config.K_zeta_bih`` unchanged whenever it is not ``None`` — that
    is the explicit pin (including ``0.0`` = the term off).

    Parameters
    ----------
    dx_mean_m : mean cell spacing of the mesh the model runs on [m]
        (``mesh.dcEdge`` averaged over real, non-padded edges).
    config : MPASOceanConfig

    Returns
    -------
    float
        The coefficient [m⁴/s] to run with.
    """
    if config.K_zeta_bih is not None:
        return config.K_zeta_bih
    if not (dx_mean_m > 0.0):
        raise ValueError(
            "resolution_scaled_k_zeta_bih: mean cell spacing must be positive, "
            f"got {dx_mean_m!r} m")
    if not (config.K_zeta_bih_ref_dx_m > 0.0):
        raise ValueError(
            "resolution_scaled_k_zeta_bih: K_zeta_bih_ref_dx_m must be positive, "
            f"got {config.K_zeta_bih_ref_dx_m!r} m")
    return config.K_zeta_bih_ref * (dx_mean_m / config.K_zeta_bih_ref_dx_m) ** 3


# A single mean spacing describes a QUASI-UNIFORM mesh.  On a mesh whose edge
# lengths span more than this factor (regional refinement), one scalar
# coefficient is over-damping the coarse part or unstable in the fine part --
# the ∇⁴ stability limit follows the SMALLEST edges -- so the derivation
# refuses rather than returning a plausible number.
K_ZETA_BIH_MAX_SPACING_RATIO: float = 4.0
