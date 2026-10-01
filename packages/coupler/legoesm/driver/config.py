"""Driver configuration dataclasses.

Provides the **canonical runtime configuration** for legoESM:

- ``ExperimentConfig`` is the single authoritative in-memory schema.
- ``GridConfig``, ``DycoreConfig``, ``OutputConfig`` are composed sub-configs.
- Native JSON serialization via ``experiment_config_to_dict`` /
  ``experiment_config_from_dict`` — no intermediate format needed.
- Backward-compatible conversion to/from the legacy
  ``AMIPExperimentConfig`` for checkpoint I/O is preserved but restricted
  to serialization boundaries (see ``to_amip_config`` / ``from_amip_config``).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, NamedTuple

from legoesm import constants

# Canonical AIMIP variant set.  Single source of truth — imported by
# ``scripts/run/run_aimip.py`` and the ``validate_strict`` rule below.
# Empty string = not an AIMIP run (preserves backward-compat for
# existing AMIP configs).
AIMIP_VARIANTS: tuple[str, ...] = (
    "", "classical", "column_nn", "sfno_physics", "sfno_full",
)


# Canonical convection-scheme set.  Single source of truth — imported by
# ``scripts/run/run_amip.py`` for its ``--convection`` CLI ``choices`` and
# used by the ``validate_strict`` membership check below.  Mirrors the
# ``integration.py`` / ``physics_pipeline`` convection factory sets, so a
# scheme added to the pipeline must be added here (and the CLI picks it up
# automatically — no second list to drift, which is exactly the bug this
# constant prevents: ``--convection tiedtke`` was rejected by a stale CLI
# ``choices`` while ``validate_strict`` accepted it).
VALID_CONVECTION_SCHEMES: tuple[str, ...] = (
    "none", "sbm", "dca", "kuo", "mass_flux", "edmf",
    "zhang_mcfarlane", "kain_fritsch", "emanuel", "tiedtke", "bechtold",
)


class GridConfig(NamedTuple):
    """Horizontal and vertical grid configuration.

    ``grid_type`` is one of the canonical names:
    ``cubed_sphere``, ``gaussian``, ``latlon``, ``mpas``.

    ``mpas`` is the SCVT Voronoi mesh + TRiSK discretization
    (Ringler 2010, Thuburn 2009).  Pre-2026-05 the codebase used
    multiple aliases for this single mesh on the atmosphere side
    (``voronoi``, ``icosahedral``, ``mpas_voronoi``) while the ocean
    consistently used ``mpas``.  All variants now normalize to
    ``mpas`` at the config boundary so the atmosphere and ocean
    use one identifier; internal dispatch checks only the canonical
    name.  See :func:`normalize_grid_type`.
    """
    grid_type: str = "cubed_sphere"  # cubed_sphere, gaussian, latlon, mpas
    resolution: int = 16             # N for CS, n_max for spectral
    nlev: int = 40
    vertical_coord: str = "hybrid"   # sigma, hybrid, cam_l32 (CAM6 32-level table)
    p_top_Pa: float = 200.0
    stretching: float = 2.0
    # B(eta) = eta**transition_exponent on the hybrid lane.  Default 3 is
    # UNCHANGED -- no committed run moves by this field existing.  It exists
    # because 3 makes the coordinate carry NEGATIVE layer mass below 663.9 hPa,
    # i.e. above ~3450 m of orography, which is 0.92% of the planet by area
    # (Tibet, the altiplano, the Greenland and Antarctic domes) and is fatal
    # where it bites: two such cells killed a 200-day idealized run in 200
    # steps (#1029).  2 moves that threshold to ~498 hPa / ~5870 m, which
    # covers all of ETOPO.  Selecting it CHANGES LEVEL PLACEMENT, so it is a
    # new baseline, not a comparison against existing runs.
    transition_exponent: int = 3
    # SIGMA-coordinate layer redistribution toward the tropopause, at FIXED
    # nlev (grids/vertical.tropopause_refined_sigma_half).  1.0 = the uniform
    # grid, bit-identical.  Uniform sigma gives ~33 hPa layers everywhere at
    # nlev=30, so the tropical tropopause is spanned by ~4 levels and no cold
    # point forms; refine=3 doubles the levels in 70-200 hPa, paid for by the
    # mid-troposphere (the lowest layer coarsens ~30%, measured).  Ignored by
    # the hybrid coordinate, which has its own `stretching`.
    tropopause_refine: float = 1.0
    # Sigma-lane lid as a fraction of surface pressure (0.01 = 10 hPa at
    # 1000 hPa); ignored by the hybrid coordinate, which uses
    # p_top_Pa/stretching.  Recorded here so the resolved config is truthful.
    sigma_top: float = 0.01
    # Centre (sigma) and log-sigma half-width of the tropopause_refine density
    # bump (grids/vertical.tropopause_refined_sigma_half).  Defaults are the
    # function's own (0.12 / 0.45 = the tropical cold point); a boundary-layer
    # refinement sets e.g. 0.95 / 0.06 (2026-09-16 arm).  Inert at
    # tropopause_refine = 1.0.
    sigma_refine: float = 0.12
    sigma_refine_width: float = 0.45
    # Sigma-lane level layout (grids.vertical.SIGMA_LAYOUTS): "standard" =
    # uniform/tropopause-refined; "l30_trop_logstrat" = the L30 troposphere
    # kept bit-identical below sigma 0.109 plus nlev-27 log-spaced
    # stratospheric layers up to sigma_top (the raised-lid experiment).
    sigma_layout: str = "standard"
    use_duogrid: bool = False        # enable FV3 Duo-Grid halo (required for MPI multi-node)


# Canonical name for the SCVT Voronoi mesh + TRiSK discretization.
# Atmosphere-side pre-2026-05 aliases that all refer to the same mesh:
_GRID_TYPE_ALIASES: dict[str, str] = {
    "voronoi": "mpas",
    "icosahedral": "mpas",
    "ico": "mpas",
    "mpas_voronoi": "mpas",
}


def normalize_grid_type(name: str) -> str:
    """Canonicalise legacy aliases for the SCVT Voronoi mesh.

    Maps ``"voronoi"``, ``"icosahedral"``, ``"ico"``,
    ``"mpas_voronoi"`` all to ``"mpas"`` (the name the ocean side
    has always used).  Every other grid_type string passes through
    unchanged.

    Callers
    -------
    * ``scripts/run_amip*.py`` argparse postprocessors.
    * Test fixtures that construct ``GridConfig`` directly with the
      legacy names.
    * Internal code that branches on grid_type SHOULD assume the
      string has already been normalised — i.e. compare to
      ``"mpas"``, not to the aliases.

    Returns
    -------
    str
        Canonical grid-type name.
    """
    return _GRID_TYPE_ALIASES.get(name, name)


class DycoreConfig(NamedTuple):
    """Dynamical core configuration."""
    model_type: str = "hydrostatic"       # shallow_water, hydrostatic, nonhydrostatic
    discretization: str = "cdgrid"        # cdgrid, spectral, sfno, u_cast, mpas
    dt: float = 600.0
    hyperdiff_scale: float = 1.0
    div_damp_scale: float = 1.0
    # Scale on the 2nd-order Laplacian viscosity A_h (see compute_diffusion).
    # The legacy A_h=0.05*dx^2/dt over-damped resolved baroclinic eddies on a
    # ~5 h timescale (faster than their ~1-2 day growth), suppressing the
    # midlatitude jets / storm tracks (dry Held-Suarez jet 2.9 vs spectral
    # 37 m/s).  a_h_scale=0 relies on the scale-selective 4th-order hyperdiff
    # alone for grid-noise control.
    a_h_scale: float = 1.0
    conservation_fixer: bool = True
    fix_mass: bool = True
    # Issue #273 Phase 3: Hoskins–Simmons FV3 D-grid implicit
    # gravity-wave damping.  When ``implicit_grav_wave_use_pcg=True``
    # and ``implicit_grav_wave_damping > 0``, the post-RK3 surface-
    # pressure correction switches from explicit forward-Euler
    # diffusion (conditionally stable at α dt / dx² < 0.5) to an
    # implicit Helmholtz solve via ``cg_helmholtz_solve`` — removing
    # the CFL ceiling on the gravity-wave-damping coefficient and
    # enabling larger production ``dt``.  Empirically supports
    # α dt / dx² up to ~50 at tol=1e-10 on a (6, n, n) cube.
    # Default OFF (False, 0.0) preserves legacy bit-exact behavior.
    implicit_grav_wave_use_pcg: bool = False
    implicit_grav_wave_damping: float = 0.0

    # Stage 3-E: Fourier polar filter for lat-lon C-grid.
    # The polar CFL problem: dx_pole = R * dlon * cos(π/2 - dlat/2) → 0
    # at the poles, forcing an explicit ``dt`` ≤ ~5 s at 1° resolution
    # even when the equatorial CFL allows ~600 s.  Enabling
    # ``use_polar_filter`` truncates Fourier modes in longitude that
    # would violate CFL at high latitudes, so the run can use the
    # equatorial-CFL ``dt`` everywhere.  Without this, 100-y AMIP at
    # 1° lat-lon FV requires ~600 B time steps and is not feasible
    # within a chained 72-h SLURM budget.
    #
    # The filter is lon-only FFT (``jnp.fft.rfft`` along axis -1), so
    # under lat-band MPI each rank applies it independently on its
    # own band — no MPI exchange needed for the filter itself.  See
    # ``polar_filter.py`` for the algorithm and
    # ``CGridLatLonPrimitiveEquationConfig.use_polar_filter`` for the
    # model-side flag this propagates to.
    use_polar_filter: bool = False
    polar_filter_cutoff_deg: float = 60.0
    polar_filter_max_wave_speed: float = 300.0

    # #836: hydrostatic lat-lon C-grid top sponge (Rayleigh damping increasing
    # toward the model lid; absorbs upward gravity-wave energy that would else
    # reflect off the rigid lid).  ``sponge_coeff=0`` (default) is OFF and
    # bit-identical.  Threaded into ``CGridLatLonPrimitiveEquationConfig`` by
    # ``component_factory`` (mirrors the polar-filter passthrough).
    sponge_coeff: float = 0.0             # Rayleigh damping scale [1/s] (exact lid
    #                                      value for 'sin2'; 'sam_rational' -> *100/101)
    sponge_width_m: float = 10000.0       # sponge-layer depth below the top [m]
    sponge_shape: str = "sin2"            # "sin2" | "sam_rational"
    sponge_scale_height_m: float = 7500.0  # log-pressure scale height for sigma->z

    # #1028: keep the cubed-sphere hydrostatic prognostic winds in FV3 D
    # staggering BETWEEN steps instead of interpolating cell-centre -> D-grid
    # corners on entry and back on exit EVERY step.  That outer round trip is
    # a measured eddy damper, not a bookkeeping detail: a paired same-commit
    # 200-day C36 Held-Suarez run (PR #1462) moved the equilibrated max wind
    # 13.0 -> 38.6 m/s (sigma) and 12.6 -> 40.9 (hybrid) with this as the only
    # change, i.e. from FAILING the #1049 dead-jet floor to clearing the
    # ~30 m/s benchmark.  The dycore has always accepted either staggering
    # (``CDGridPrimitiveEquationModel.step`` dispatches on the state type);
    # what was missing was a production driver that carries the D state.
    #
    # Cubed-sphere hydrostatic ONLY.  Every other lane (lat-lon, MPAS,
    # spectral, shallow water, non-hydrostatic) and the lanes this does not
    # cover yet (tiled / sub-face SPMD, ensembles) REFUSE it loudly rather
    # than silently running the damped path.  Default False keeps every
    # existing run byte-identical.
    persistent_dgrid: bool = False


    # Task #25: time integrator override.  Lat-lon C-grid uses
    # ``ssp_rk3`` by default — three RK3 stages unrolled with the
    # tendency function inlined 3×.  Setting
    # ``time_integrator="ssp_rk3_scan"`` folds the 3 stages into a
    # single ``jax.lax.scan`` body so XLA optimises the tendency
    # pipeline ONCE.  Same SSP coefficients (α = (0, 0.75, 1/3),
    # β = (1, 0.25, 2/3)), same number of tendency calls per step,
    # IEEE-identical output (pinned by
    # tests/timestepping/test_ssp_rk3_scan_bit_equivalence.py).
    # At the production AMIP shape (lat-lon C-grid + tracers + polar
    # filter) profile job 8070275 measured a 1.3–1.9× JIT compile
    # speedup — meaningful for the 100-y AMIP submission where the
    # smoke jobs were paying ~2.5 h of compile per rank-count.
    #
    # On cube / lat-lon, ``"auto"`` resolves to ``ssp_rk3`` — bit-
    # equivalent behaviour for the existing scientific validation
    # suite.  ``"ssp_rk3_scan"`` is the opt-in for production at scale.
    #
    # ``"auto"`` (the default here AND the run_amip CLI default) selects
    # each dycore's own stable default in ``component_factory``:
    # cube / lat-lon keep ``ssp_rk3``; MPAS gets ``ssp_rk54_scan``
    # (its biharmonic hyperdiffusion eigenvalues at production dt fall
    # outside the ssp_rk3 stability region — the former "hidden CFL"
    # blow-up); spectral keeps ``ssp_rk54``.  Any explicit scheme name
    # (including ``ssp_rk3`` on MPAS) is forwarded verbatim, so
    # deliberate integrator-sensitivity runs are still possible.
    # Default flipped ``"ssp_rk3"`` → ``"auto"`` (2026-07-12): a direct
    # ``DycoreConfig()`` on MPAS previously inherited the documented-
    # unstable ssp_rk3 (diverges within ~3 steps at dt=600 with
    # hyperdiff ON) — only the run_amip CLI got the safe per-dycore
    # resolution.  ``"auto"`` never reaches ``dispatch_integrator``
    # (every factory branch maps it first; dispatch raises loudly on
    # unknown names as defense in depth).
    time_integrator: str = "auto"
    # #771: transport the (attached) moisture tracers horizontally with the
    # mass-conserving flux-form post-RK3 substep instead of the in-RK3 advective
    # -(u·∇q).  Fixes the cube column-water non-conservation / day-150 blow-up.
    # Only takes effect on the cubed_sphere cdgrid PE dycore AND with moisture
    # attached (``ExperimentConfig.moisture_advection=True``).  EXPERIMENTAL,
    # default off (advective path bit-exact); serial-only (fail-closed under MPI
    # face-scatter until the reductions are allreduce-aware).  Appended last to
    # preserve positional ABI.
    moisture_flux_form: bool = False
    # #930: vertical biharmonic (∂⁴/∂σ⁴) hyperdiffusion coefficient [1/s] for T
    # on the MPAS hydrostatic dycore — scale-selective damping of the grid-scale
    # 2Δσ vertical checkerboard that the adiabatic κ·T·ω/p term amplifies (no
    # other vertical operator in that dycore opposes it) until it rides the
    # silent T_min=50 K floor (#871/#912/#915).  del4 damps 2Δσ ~47× faster
    # than an 8Δσ resolved wave, so resolved vertical structure is ~untouched;
    # explicit-stable to huge dt (16·ν·dt≪1).  Only wired to the MPAS PE dycore
    # (``component_factory``).  Set 0.0 to reproduce the pre-#930 dycore exactly.
    # Appended last to preserve positional ABI.
    mpas_nu_vert4_T: float = 2.0e-6
    # Column-CONSERVING tracer positivity clamp in the MPAS floors stage.
    # The plain ``max(q, 0)`` is NOT mass-neutral: with no limiter in
    # ``tracer_transport_mpas``, horizontal advection undershoot alone made it
    # invent +0.0822 kg/m2/day (+30 kg/m2/yr) of water on the AMIP century
    # (measured 2026-07-26, 96% from q_i/q_c), driving column water 23->42
    # kg/m2, OLR 199->109 W/m2 and +10 K/yr warming.  True borrows the clipped
    # deficit back from the positives (10.4x less spurious mass, measured).
    # Default TRUE since 2026-08-16 (owner decision: "conserving form
    # always"); ``--no-mpas-conservative-tracer-clamp`` restores the legacy
    # mass-creating clamp for bit-comparison against older runs.
    # #1354/#1515: this knob is now GRID-GENERAL — it also drives the borrow on
    # the cube, spectral and lat-lon lanes (each previously had a plain clamp or
    # NO floor at all), so every grid's tracer positivity is equivalent.  The
    # ``mpas_`` prefix is legacy; a rename is deferred to avoid a schema churn.
    mpas_conservative_tracer_clamp: bool = True
    # #1029 ω-side: SB81 α-weighted κT·ω/p energy conversion on the hybrid
    # lat-lon C-grid lane (discretization-consistent with the geopotential
    # and the momentum/thermo ln p^SB gradients).  Default OFF — the
    # consistent form removes the arithmetic form's accidental damping of
    # the #1029(b) lid-amplified orographic-wave mode (held_suarez_topo
    # latlon blowup day ~49 -> ~12, A/B job 9130802); opt-in until the lid
    # treatment lands.  Threaded by ``component_factory`` (mirrors the
    # sponge/polar-filter passthrough).  Appended last to preserve
    # positional ABI (codex #1029 r3 #2).
    sb81_omega_conversion: bool = False
    # Separate scale for the horizontal THERMAL diffusivity K_h (None = follow
    # a_h_scale exactly as before, byte-identical).  Decouples the circulation
    # lever (momentum nu_del2) from the thermal smoothing that damps vertical
    # computational modes.  WIRED ONLY into the MPAS thermal diffusion path;
    # validate_strict refuses it on other discretizations (silently-inert
    # guard, same pattern as hard_sat_ice_curve).  Appended at the tuple END
    # to preserve the positional ABI (codex review).
    k_h_scale: float | None = None
    # Vertical advection scheme on the MPAS SIGMA lane ("upwind" | "van_leer").
    # First-order upwind's implicit diffusion K_σ = |σ̇|·Δσ/2 is +0.822 K/day at
    # the tropical UTLS maximum (15S-15N, 91.4 hPa, cldF_fsd, N=37) — larger
    # than the whole production temperature tendency there and 2.1x the
    # radiative cooling.  "van_leer" is the 2nd-order TVD alternative: bounded
    # face reconstruction, monotone update under a Courant condition
    # ~ nu_k + nu_{k+1} <= 1 DERIVED FOR A UNIFORM GRID (stretched grids and
    # varying sigma_dot have regression evidence only — see the kernel
    # docstring).  Measured global max of that pair on that run: 0.0642, 15.6x
    # inside the uniform-grid bound, from 37 checkpoint snapshots.
    # Requires nlev >= 4.
    # WIRED ONLY into the MPAS sigma lane; validate_strict refuses it on other
    # discretizations / vertical coordinates rather than let it run silently
    # inert.  Default "upwind" keeps every existing result bit-identical.
    # Appended at the tuple END: preserves POSITIONAL CONSTRUCTION by existing
    # callers, not full tuple ABI (exact unpacking / len() still break).
    mpas_vert_advection_scheme: str = "upwind"
    # MPAS lane, CAM-style top diffusion sponge: del2 viscosity x factor**((n-k)/n)
    # in the top n layers (raised-lid experiment); 0 / 1.0 = off.
    mpas_sponge_del2_top_layers: int = 0
    mpas_sponge_del2_top_factor: float = 1.0
    # FV3 duo window SPMD (M6): split every face into kt x kt windows, one
    # per device (6*kt*kt ranks).  Both EXPLICIT, no auto-selection and no
    # default pad: the pad is a measured, per-deck halo width (11 at C48
    # with 3 acoustic substeps), not a formula, so it must appear in the
    # run's config (user call 2026-09-21).  None = the face layout.
    # MERGE NOTE: appended AFTER main's sponge pair, so main's positional
    # layout is untouched and these two land at the new tuple end -- the same
    # invariant both sides were preserving independently.
    fv3_duo_windows: int | None = None
    fv3_duo_window_pad: int | None = None
    # FV3 duo as a COLUMN model inside the MPAS lane (route A, 2026-09-26,
    # docs/architecture/fv3_duo_amip_adapter_plan.md): the duo is the
    # dynamics operator of ``_run_mpas`` through FV3DuoColumnModel, so the
    # CAM6 AMIP suite (physics on (nCells, nlev) columns) drives it
    # without any physics rewrite.  False = the closed certified duo lane.
    fv3_duo_column_lane: bool = False
    # FV3's own tracer positivity in the vertical remap (fv_mapz.F90 fill
    # -> fillz column borrow, fv_fill.F90).  Default False = the certified
    # oracle deck (input.nml fill=.F.); the CAM6 deck sets it True
    # (decision B3, 2026-10-01: every-step repair cadence as on MPAS).
    fv3_duo_fill: bool = False

    # Divergence-SELECTIVE biharmonic damping on the MPAS hydrostatic lane,
    # as a multiple of CAM-FV's own ldiv4 coefficient 0.01*area^2/dt
    # (cd_core.F90:655-683; fv_div24del2flag=4 is the CAM6 physics default at
    # every horizontal grid).  1.0 = CAM's strength, 0.0 = off and
    # byte-identical.  The existing nu_del2/nu_del4 are VECTOR Laplacians and
    # damp balanced flow together with the divergent mode; this term does not.
    # Appended at the tuple END: preserves POSITIONAL CONSTRUCTION.
    mpas_div_damp4_scale: float = 0.0


class EvaluationConfig(NamedTuple):
    """Post-run ClimateEval configuration.

    When ``enabled=True`` and ``cmip_output=True``, the driver invokes
    ClimateEval (via ``climateeval_hook.maybe_run_climateeval``) after a
    successful AMIP run, comparing CMOR outputs against ERA5 /
    observational reference data. ClimateEval is a separate, externally
    installed tool (github.com/climate-federation/ClimateEval) — NEVER a
    legoESM dependency (its iris/ESMValTool stack is heavy/conda-only and
    conflicts with the JAX environment). The hook shells out to
    ``climateeval_python`` rather than importing ``climateeval`` into this
    process. ``suites`` are pass-through names consumed by that external
    tool (not legoESM scheme dispatch), so they are intentionally not
    membership-validated here; an unknown suite fails loudly inside
    ClimateEval's own ``Suite()`` constructor. All listed suites are
    rendered into ONE combined HTML report. **Empty (the default) = run
    ALL bundled suites (every tier)**; a suite whose data is missing /
    inapplicable is skipped and reported by the runner, not fatal.

    ``climateeval_python`` and ``data_root_dir`` have no hardcoded
    personal defaults — the CLI (``run_amip.py --evaluation-climateeval-
    python`` / ``--evaluation-data-root-dir``) defaults them from the
    ``LEGOESM_CLIMATEEVAL_PYTHON`` / ``LEGOESM_CLIMATEEVAL_DATA_ROOT``
    environment variables instead, since both paths are inherently
    per-user/per-machine (there is no shared, canonical install or
    reference-data location). If either is left unset while
    ``enabled=True``, ``ExperimentConfig.validate_strict`` raises LOUDLY
    before the run starts (see setup instructions in
    ``docs/user-guide/climateeval_evaluation.md``).
    """
    enabled: bool = False
    suites: tuple[str, ...] = ()  # empty = ALL bundled suites (every tier)
    model_id: str = "legoESM-1-0"
    experiment_id: str = "amip"
    variant_id: str = "r1i1p1f1"
    data_root_dir: str = ""
    fail_on_missing_data: bool = False
    download_missing_data: bool = False
    timerange: str = ""
    climateeval_python: str = ""


class OutputConfig(NamedTuple):
    """Output and diagnostics configuration."""
    output_dir: str = ""
    diag_days: float = 5.0   # diagnostic cadence [days]; sub-daily values are honoured (a NaN is reported at the first sample, so a coarse cadence only bounds it from above)
    checkpoint_days: int = 0
    monthly_means: bool = False
    cmip_output: bool = False
    clear_sky_diag: bool = False
    # Per-process column water/energy budget ledger (diagnostics.
    # process_ledger): attributes the global column-store tendencies to
    # turbulence/convection/microphysics/radiation/other/clips/dynamics
    # every step and writes segment-mean rates to budget_ledger.npz.
    # Static diagnostic gate (default OFF = byte-identical model);
    # single-rank only.
    budget_ledger: bool = False
    # Restrict every ledger row to a sigma band (lo, hi), lo < hi in [0, 1] and
    # sigma increasing downward.  None = the whole column.  WHY: the column
    # ledger is structurally blind to a vertical-REDISTRIBUTION bias, because a
    # mass-flux convection scheme's column water row is exactly zero (measured:
    # 0.0000 kg/m2/day on the production AMIP) while the tropical free
    # troposphere is twice as moist as observed.  Banding is what makes the
    # table answer "which process supplies THIS layer", without changing the
    # shape of the accumulator carried through the jitted step.
    budget_ledger_sigma_band: tuple | None = None
    checkpoint_format: str = "npz"  # npz, zarr
    diagnostics_perf_mode: str = "auto"  # auto, always, never
    cmip_resolution_deg: float = 5.0  # lat-lon grid spacing for CMIP output [degrees]
    # Wallclock-aware mid-job checkpointing for long (century-scale) HPC chains.
    # ``max_wallclock_seconds=0`` disables it (default); set it to the SLURM
    # ``--time`` budget so the run checkpoints and exits cleanly with
    # ``restart_buffer_seconds`` to spare, letting a dependency chain resume.
    max_wallclock_seconds: float = 0.0
    restart_buffer_seconds: float = 600.0
    evaluation: EvaluationConfig = EvaluationConfig()


# Single source of truth for the valid column-physics scheme literals — consumed
# by ExperimentConfig.validate_strict AND by run-driver CLI ``choices=`` so the
# CLI allowlist cannot drift from the config validation (e.g. omitting an
# advertised scheme like ``ml_emulator``).
#
# These were function-local variables inside validate_strict, so nothing could
# import them and every driver kept its own hand-copied list — which is exactly
# why they drifted: each axis happened to be pinned by a point-fix test in ONE
# driver and silently diverged in the other (run_coupled blocked bechtold/
# tiedtke/emanuel/kain_fritsch/zhang_mcfarlane; run_amip blocked mynn25).
# ``tests/unit/test_scheme_reachability_audit.py`` now machine-audits every
# driver's ``choices=`` against these tuples, against a shrink-only baseline of
# deliberate exclusions.
VALID_MICROPHYSICS = (
    "none", "kessler", "sundqvist", "seifert_beheng",
    "morrison", "thompson", "p3", "sdm", "fast_sbm", "ml_emulator",
)

VALID_TURBULENCE = (
    "smagorinsky", "louis", "tke", "mynn25", "clubb_lite", "clubb",
    "holtslag_boville", "ysu", "edmf", "none",
)

VALID_RADIATION = ("none", "gray", "rrtmgp", "rrtmg")

VALID_CLOUD_SCHEMES = ("none", "sundqvist", "xu_randall", "resolved", "cam6_clubb")

# ``gravity_wave_drag`` additionally accepts a ``+``-joined COMPOSITION of these
# (e.g. "hines+mcfarlane"); validate_strict splits on "+" before membership.
VALID_GWD = (
    "rayleigh", "lindzen", "mcfarlane", "hines",
    "prognostic_spectral", "e3sm_cam", "ml_emulator", "none",
)

def parse_gwd_spec(value: str) -> str:
    """argparse ``type=`` for ``--gravity-wave-drag``.

    GWD is the one axis a plain ``choices=`` CANNOT express: it accepts a
    ``+``-joined COMPOSITION whose source tendencies are summed (issue #834,
    e.g. "hines+mcfarlane"), because orographic and non-orographic drag
    parameterize distinct wave populations and are run together in CMIP-class
    GCMs.  run_amip therefore dropped ``choices`` entirely -- which left the
    flag with NO cli-level typo rejection -- while run_coupled kept ``choices``
    and so REJECTED every composite, making #834 unreachable from the coupled
    driver.  Both drivers now share this validator: composites work everywhere,
    and a typo is still caught at the CLI.

    Delegates the SEMANTICS to ``ExperimentConfig.validate_strict`` rather than
    re-implementing them: a membership-only check accepted composites strict
    rejects -- "none+hines", "ml_emulator+hines", duplicates
    like "hines+hines" (codex) -- so the CLI would advertise a spec the config
    then refuses. Asking the real validator keeps the two from diverging by
    construction, which is the same lesson as resolving the effective surface
    config through the production path in driver/air_sea_consistency.py.
    """
    import argparse

    for part in value.split("+"):
        if part not in VALID_GWD:
            raise argparse.ArgumentTypeError(
                f"invalid gravity-wave-drag source {part!r} in {value!r}; "
                f"expected one of {VALID_GWD}, or a '+'-joined composite of "
                f"them (e.g. 'hines+mcfarlane')"
            )
    try:
        # The e3sm_cam-in-composite rule depends on --e3sm-cam-source, parsed
        # separately; probe with the one source that composes so the real
        # gate fires at build time with the actual value, not here.
        _parts = value.split("+")
        _probe = ({"e3sm_cam_source": "background", "e3sm_cam_pgwv": 1}
                  if "e3sm_cam" in _parts and len(_parts) > 1 else {})
        ExperimentConfig(gravity_wave_drag=value, **_probe).validate_strict()
    except ValueError as exc:
        # Report ONLY a genuine gravity_wave_drag complaint. validate_strict
        # reports every error for the whole config, so falling back to the full
        # message would blame this flag for an unrelated bad default elsewhere
        # (codex). If nothing here is about GWD, this value is not the problem
        # -- let it through and let the config's own validation report the real
        # error, in its own words, at build time.
        msg = "; ".join(m for m in str(exc).splitlines()
                        if "gravity_wave_drag" in m)
        if msg:
            raise argparse.ArgumentTypeError(msg) from None
    return value


# The ATMOSPHERE surface layer's bulk-flux algorithm.  "most" is deliberately
# ABSENT: turbulence/surface_layer.py dispatches MOST on ("coare3",
# "large_yeager") only, so an accepted "most" would silently degrade to the
# constant-coefficient branch — a loud rejection is better than wrong physics.
# See driver/air_sea_consistency.py.
VALID_SURFACE_BULK = ("constant", "coare3", "large_yeager", "large_yeager_cesm")


# Hard ceiling of ``morrison_sed_cfl_substeps_max``, imported from the leaf
# config that owns the field so the two validators cannot drift apart (GLM
# 2026-09-22).  The loop cost is LINEAR in the cap, so a typo ("2560")
# silently decouples a run's cost from its physics; the largest configured
# lane needs 269 (CAM L32 at 1800 s).
def _sed_substeps_cap_limit() -> int:
    """The leaf config's hard ceiling (deferred import: this module must not
    pull the atmosphere package at import time)."""
    from legoesm.atmosphere.physics.microphysics.config import (
        SED_CFL_SUBSTEPS_MAX_LIMIT,
    )
    return SED_CFL_SUBSTEPS_MAX_LIMIT


class ExperimentConfig(NamedTuple):
    """Top-level experiment configuration.

    This is the **canonical runtime schema** for legoESM.  All driver,
    physics, forcing, and diagnostic code should consume this type.

    Composes GridConfig, DycoreConfig, OutputConfig with physics
    and forcing parameters.
    """
    grid: GridConfig = GridConfig()
    dycore: DycoreConfig = DycoreConfig()
    output: OutputConfig = OutputConfig()

    # Integration
    days: int = 200
    start_day: float = 0.0
    # Optional seasonal alignment for the radiation insolation ONLY (decoupled
    # from the relative-indexed AMIP SST forcing). When set, model day 0 maps to
    # this noleap day-of-year [1, 366) for the insolation day_of_year, so an
    # AMIP run started from a non-January ERA5 date can run the matching solar
    # season WITHOUT shifting start_day (which would push the relative SST out of
    # range). None => legacy behavior: day 0 -> Jan 1 (day_to_calendar(0)).
    # See docs/COMPARE_REANALYSIS.md (iter 449). CODEX PENDING (radiation path).
    insolation_start_doy: float | None = None

    # Forcing
    dataset: str = "analytical"
    forcing_path: str = ""
    sic_path: str = ""             # optional separate SIC file
    sst_var: str = ""
    sic_var: str = ""
    time_var: str = ""
    lat_var: str = ""
    lon_var: str = ""
    sst_offset: float = 0.0
    sic_scale: float = 1.0

    # Radiation
    radiation: str = "gray"
    rad_update_steps: int = 1
    # Physics cadence (CAM6 suite): run the WHOLE column physics every N
    # steps and re-apply its cached tendency set (u, v, T, p_s, every
    # tracer, precip / surface / TOA diagnostics, ledger rows) on the steps
    # in between -- CAM's physics-every-1800-s pattern.  1 = every step
    # (byte-identical to before).  MPAS lane only; ``rad_update_steps`` must
    # be a multiple of it so radiation stays on physics steps.
    physics_update_steps: int = 1
    # Un-fuse radiation from the compiled-segment scan (issue: ~3h XLA
    # compile).  Static Python gate (NOT trainable); default OFF keeps
    # every existing run byte-identical.  When True AND
    # ``rad_update_steps > 1`` the PRODUCTION ``_run_compiled`` path lifts
    # the radiation-cycle loop from XLA to the host so rrtmgp and the
    # dynamics+physics scan compile as TWO SEPARATE executables (a jit
    # placed inside a ``lax.scan`` is inlined by XLA, not a distinct
    # compile unit — only a host-level jit is its own executable).
    unfused_radiation: bool = False
    diurnal_cycle: bool = False
    # Realistic (Berger 1978) orbital insolation for AMIP-II / CMIP.  When
    # True the radiation uses the present-day orbital declination and scales
    # TOA insolation by the Earth-Sun distance factor (a/r)^2 (eccentricity
    # perihelion/aphelion asymmetry, ~+/-3.4%).  Default False keeps the
    # circular-orbit approximation for idealized/aquaplanet runs.
    orbital_insolation: bool = False
    # RRTMGP column recurrence implementation:
    #   False = Python for-loop (fully unrolled XLA graph, GPU-friendly default)
    #   True  = jax.lax.scan (smaller graph, often slower per step on GPU but
    #           reduces compile time and is preferred for large nlev or AD)
    # Issue #273 GPU tuning: ``None`` defers the choice to
    # ``rte_utils.recurrent_op_with_halos`` which auto-picks
    # ``True`` on GPU/TPU (collapses ``nlev`` separate kernel
    # launches into one fused ``lax.scan`` — the biggest single win
    # against the 2600s cold-compile time called out in issue #273)
    # and ``False`` on CPU.  Explicit ``True``/``False`` overrides.
    rrtmgp_use_scan: bool | None = None
    # G-point parallelism in the RRTMGP two-stream solve (see
    # ``RRTMGPConfig.gpoint_batch_size``).  DEFAULT 16 (was 0): the vmap-block
    # path compiles ONE reused body (the scan path's per-g-point prevent_cse
    # body inflated the reverse-mode-AD compile to ~9 h) and runs ~26x faster,
    # with peak memory bounded to this many g-points.  Set 0 only to reproduce
    # the exact legacy g-point accumulation order.
    rrtmgp_gpoint_batch_size: int = 16
    # G-point checkpointing in the RRTMGP two-stream scan (see
    # ``RRTMGPConfig.gpoint_checkpoint``).  True (default) = ``jax.checkpoint``
    # with ``prevent_cse=True`` per g-point — memory-frugal, REQUIRED for
    # reverse-mode AD / training.  False = plain ``lax.scan`` (no prevent_cse):
    # smaller compiled footprint / faster cold compile for FORWARD/inference
    # runs, used to relieve the XLA-CPU LLVM-JIT code-region pressure.
    rrtmgp_gpoint_checkpoint: bool = True
    # Column-chunk the rrtmgp solve to cap the XLA compile time at higher
    # horizontal resolution (see ``RRTMGPConfig.column_chunk_size``).  0 =
    # off (byte-identical). >0 = jax.lax.map the solve over fixed-size column
    # blocks; the per-block body compiles ONCE at this size (columns are
    # independent → numerically exact; must divide the column count).
    rrtmgp_column_chunk_size: int = 0
    co2_ppmv: float = 415.0
    ch4_ppbv: float = 1900.0
    n2o_ppbv: float = 332.0
    S_0: float = constants.S_0
    ozone_source: str = "standard"
    ozone_forcing: str = "inline"       # inline, external, off
    ozone_file: str = ""
    ghg_forcing: str = "constant"       # constant, external
    ghg_file: str = ""

    # Solar
    solar_source: str = "constant"      # constant, file, spectral_file
    solar_file: str = ""
    solar_tsi_var: str = "tsi"
    solar_spectral_var: str = "solar_fraction_by_gpt"
    solar_spectral_band_order: str = "auto"   # auto | as_is | rrtmg_sw (#322)

    # Aerosol
    aerosol_forcing: str = "off"        # off, external
    aerosol_file: str = ""
    aerosol_reference_aod: float = 0.03
    volcanic_aerosol_file: str = ""
    volcanic_aerosol_scale: float = 1.0
    # Volcanic stratospheric LONGWAVE aerosol (gap #9): when True, ALSO
    # load the ``ext_earth`` LW band from ``volcanic_aerosol_file`` and
    # thread it into RRTMGP as the LW absorption optical depth.  Default
    # OFF ⇒ no LW aerosol (byte-identical; zeros LW od is a solver no-op).
    volcanic_aerosol_lw: bool = False

    # Clouds & Microphysics
    cloud_scheme: str = "none"
    # Route a moist higher-order turbulence closure's (CLUBB) sub-grid PDF cloud
    # fraction into the cloud optics instead of the RH grid-scale one — the
    # marine-Sc over-bright albedo lever.  Maps to
    # ``RadiationConfig.use_clubb_cloud_fraction``; requires diagnostic CLUBB
    # turbulence (turbulence='clubb').  False (default) is byte-identical.
    use_clubb_cloud_fraction: bool = False
    # Run CLUBB as a PROGNOSTIC higher-order closure rather than a diagnostic
    # one: the scheme then carries 15 higher-order moments as real state,
    # including the total-water variance, the liquid-water-potential-temperature
    # variance and their covariance.  Diagnostic CLUBB instead estimates those
    # variances each step from a mixing length times a local gradient, so the
    # cloud PDF is handed an equilibrium guess rather than a quantity with
    # memory.  Requires turbulence='clubb'; ``init_physics_state`` seeds the
    # packed (ncol, 15, nlev+1) carry when this is on, and the turbulence
    # dispatch refuses to run prognostic CLUBB against an unseeded carry rather
    # than silently re-seeding it every step.  False (default) is the
    # byte-identical diagnostic path.
    clubb_prognostic: bool = False
    # Whether CLUBB exchanges CLOUD LIQUID with the host, as CAM's
    # ``clubb_intr.F90`` does (rt = q_v + q_c in, q_v = rt - rcm and
    # q_c := rcm out).  False (default) keeps the historical bridge, which
    # hands the advanced total water back wholly as vapour and so never gives
    # the host the liquid the closure's own PDF diagnosed -- the host then
    # takes cloud FRACTION from CLUBB and cloud WATER from a tracer CLUBB never
    # wrote.  Requires turbulence='clubb' and the prognostic path, plus a
    # condensate tracer to write into; the MPAS lane is the only one wired to
    # route the liquid tendency, and the others refuse rather than drop it.
    # Turning this on MOVES water between two host tracers and changes the
    # cloud radiative state, so it is a prognostic change, not a diagnostic one.
    clubb_liquid_partition: bool = False
    # CLUBB's upper domain limit [Pa] (CAM ``trop_cloud_top_press``): the
    # scheme's mixing is tapered to zero above this pressure.  None (default)
    # keeps the scheme's own 0.0 = no limit, byte-identical.
    clubb_trop_cloud_top_press: float | None = None
    # Moisture-only multiplier on CLUBB's q_v eddy diffusivity at the faces
    # whose sigma lies in ``clubb_q_flux_scale_sigma_band`` (cloud-base mixing
    # probe, 2026-09-20).  None (default) keeps the scheme's own 1.0,
    # byte-identical.  Diagnostic CLUBB only.
    clubb_q_flux_scale: float | None = None
    clubb_q_flux_scale_sigma_band: tuple[float, float] | None = None
    # Opt-in convective (cumulus) cloud-fraction source (Slingo-1987-inspired surrogate).  The
    # RH-based stratiform cloud schemes give ~0 cloud where an adjustment
    # convection scheme (sbm) holds the column subsaturated, so the convecting
    # tropics radiate surface LW to space (~4.5 K coupled cold bias).  When
    # True, the convective precip rate drives a bounded cumulus cover in the
    # cloud diagnosis (see CloudConfig.convective_cloud).  Default False =>
    # byte-identical to the validated stratiform-only path.
    convective_cloud: bool = False
    # Optional cloud-tuning overrides for the diagnostic stratiform/convective
    # cloud (None => CloudConfig defaults => byte-identical).  Exposed so a
    # coupled run can trade SW (planetary albedo) vs LW (greenhouse / surface
    # LW_down) without editing CloudConfig in source:
    #   cloud_rh_crit        — Sundqvist critical RH; HIGHER => less stratiform
    #                          cloud (lower albedo).  Bounds (0.5, 0.99).
    #   cloud_q_c_diagnostic — diagnostic in-cloud condensate [kg/kg]; LOWER =>
    #                          optically THINNER cloud (lower albedo, still
    #                          LW-active).  Bounds (1e-6, 1e-3) — the lower end
    #                          was widened from 5e-5; see validate_strict.
    #   cloud_conv_cloud_max — convective (Slingo-1987-inspired surrogate) cover cap.  Bounds (0.1, 1.0).
    #   cloud_conv_cloud_condensate — convective anvil in-cloud condensate
    #                          [kg/kg]; LOWER => optically THINNER / more realistic
    #                          anvil (lower albedo, still LW-active).  Bounds
    #                          (1e-5, 1e-3).
    # These are the SW/LW knob for the coare3 moisture-driven albedo overshoot.
    cloud_rh_crit: float | None = None
    cloud_q_c_diagnostic: float | None = None
    # Cahalan (1994) horizontal-inhomogeneity factor chi on the radiative cloud
    # water path (plane-parallel albedo bias); LOWER => thinner optics => lower
    # albedo. None => CloudConfig default 1.0 (homogeneous, no change).
    cloud_inhomogeneity_factor: float | None = None
    # Sub-grid cloud-optics inhomogeneity scheme: "constant" (Cahalan scalar,
    # legacy/byte-identical) or "two_region" (tau-dependent Shonk-Hogan optic
    # that breaks the plane-parallel tau-saturation). cloud_fsd = fractional
    # std-dev of in-cloud water for two_region (Shonk-Hogan ~0.75).
    cloud_optics_inhomogeneity: str = "constant"
    cloud_fsd: float | None = None
    # Partial-cloud-COVER optics: "none" (legacy/byte-identical) or
    # "two_column".  The solver has no McICA/overlap and sees ONE
    # homogeneous column at the grid-mean path, i.e. R(cf*tau_ic); the
    # independent-column answer cf*R(tau_ic)+(1-cf)*R(0) is DARKER because R
    # is concave.  "two_column" applies the exact inversion of that identity.
    cloud_partial_coverage_optics: str = "none"
    # VERTICAL overlap optics: "none" (legacy/byte-identical) or
    # "max_random" (n_sub deterministic maximum-random-overlap subcolumns,
    # measured -30% cloud albedo and +18 W/m2 OLR vs a Monte-Carlo
    # reference; costs n_sub x the radiation time). MUTUALLY EXCLUSIVE with
    # cloud_partial_coverage_optics="two_column" -- both correct partial
    # coverage, so enabling both double-discounts the cloud.
    cloud_vertical_overlap_optics: str = "none"
    cloud_n_subcolumns: int = 8
    # Saturation curve for the cloud-fraction RH (CloudConfig.saturation_scheme):
    # "mixed_phase" (DEFAULT since 2026-09-17: RH against the
    # ice-fraction-blended liquid/ice curve, IFS alpha(T) convention — so
    # ice-saturated TTL/anvil/POLAR air reads RH ~1 and the RH cloud schemes see
    # the cirrus the model already carries, #1521) or "liquid" (legacy, liquid
    # Tetens saturation at all T; byte-identical reproduction of pre-2026-09-17
    # runs only).
    #
    # Why the default moved, and it is a defect report rather than a preference:
    # cover is zero below rh_crit (0.85 in production) and "liquid" measured RH
    # against the LIQUID curve at every temperature.  At 230 K and 900 hPa,
    # ice-saturated air has RH_liquid = 0.662, so reaching 0.85 needs ~28 % ice
    # SUPERsaturation — Arctic cloud was arithmetically impossible, and the
    # measured February cover north of 80N was 0.0 % against 35-40 % observed.
    # Zero cover then also removes EXPLICIT ice condensate from the radiative
    # subcolumns, so the model's own cirrus was radiatively invisible there.
    cloud_saturation_scheme: str = "mixed_phase"
    #   cloud_p_xr / cloud_alpha_xr — Xu-Randall cloud-fraction sensitivity
    #   knobs; HIGHER p_xr / LOWER alpha_xr => fraction stays fractional as
    #   moisture rises (flattens the overcast runaway).
    cloud_p_xr: float | None = None
    cloud_alpha_xr: float | None = None
    # Condensate-aware cover floor cf >= q_cond/(q_cond + q_ref) [kg/kg] for the
    # RH-diagnosed schemes (CloudConfig.cover_condensate_q_ref).  None => scheme
    # default (0.0 = off).  Paired-arm lever for the invisible-ice defect.
    cloud_cover_condensate_q_ref: float | None = None
    # Polar-cap radiative cloud floor (CloudConfig.cap_floor_*): an attribution
    # lever for the 2026-09 Arctic self-isolation A/B, radiation-only, MPAS /
    # spectral standalone radiation path only (the FV pipeline refuses it).
    # Off in production; None on the floats => CloudConfig defaults.
    cloud_cap_floor_on: bool = False
    cloud_cap_floor_lat_deg: float | None = None
    cloud_cap_floor_p_max_pa: float | None = None
    cloud_cap_floor_cf: float | None = None
    cloud_cap_floor_q_c: float | None = None
    # Snow grain-growth activation temperature [K] (BATS ~5000): the snow-age
    # clock accumulates dt*exp(A*(1/T_freeze - 1/T_snow)) so cold dry snow keeps
    # its fresh albedo. None => LandAlbedoConfig default (0.0 = off, the
    # calendar clock, byte-identical).
    snow_age_activation_K: float | None = None
    # Snow-albedo age e-folding time [days].  None => the calibration's value
    # (3.674 d under land_calibrated_physics, 11.64 d otherwise).
    # WHY THIS IS A KNOB AT ALL: measured 2026-09-11 on the production AMIP,
    # every snow-covered polar cell sat at the fully-aged albedo 0.521 against
    # an observed 0.70 (Arctic tundra) to 0.82 (Antarctic plateau), because a
    # 3.7-day clock darkens anything older than a few weeks regardless of how
    # cold it is.  Turning on the temperature dependence alone does NOT fix it:
    # an arm at the BATS activation of 5000 K slowed the clock by only 5-12% in
    # ten days and left the albedo unchanged at 0.521.  The e-folding time is
    # the binding parameter, and it was previously a hardcoded calibrated
    # constant no run could select.  NOTE the calibration fitted 3.674 d with
    # the temperature dependence OFF, so the two should eventually be refitted
    # together; an arm moving this alone trades against that land-surface
    # temperature calibration and must be scored on both.
    # DEFAULT 150 d (user 2026-09-15) once the canopy's absorbed shortwave and
    # the exported albedo were reconciled: measured on a 30-day pair together
    # with albedo_ice 0.80, polar clear-sky reflected SW bias -15.7 -> -2.7
    # W/m2. None restores the calibration's 3.674 d.
    land_snow_tau_days: float | None = 150.0
    # Marine-Sc albedo lever: blend strength [0,1] toward diagnostic-CLUBB cf in
    # the BL when --use-clubb-cloud-fraction is on (1.0 = full replacement, which
    # drove a real-SST surface-heating runaway; ~0.3-0.5 is gentler + stable).
    # None => CloudConfig default (1.0).
    cloud_clubb_cf_override_strength: float | None = None
    # Marine-Sc lever cloud-collapse floor [0,1]: minimum BL cloud the override
    # may leave (breaks the cloud-temperature runaway that full reduction caused).
    # None => CloudConfig default (0.0 = no floor).
    cloud_clubb_cf_override_floor: float | None = None
    cloud_conv_cloud_max: float | None = None
    cloud_conv_cloud_condensate: float | None = None
    # Diagnostic in-cloud condensate vertical structure for the stratiform
    # radiative floor (CloudConfig.diagnostic_condensate_scheme):
    #   "constant"  — flat q_c_diagnostic at every cloudy level (validated
    #                 default; byte-identical to the legacy floor).
    #   "adiabatic" — depth-scaled adiabatic in-cloud LWC (dims THIN warm
    #                 marine stratocumulus while deep clouds stay at the cap),
    #                 the source-side marine-BL albedo fix.  cloud_adiabatic_lwc_rate
    #                 [kg/kg/m] is the LWC growth per metre of cloudy depth
    #                 (None => CloudConfig default 1.5e-6 ~ 1.5 g/kg per km).
    cloud_diagnostic_condensate_scheme: str = "constant"
    cloud_adiabatic_lwc_rate: float | None = None
    microphysics: str = "none"
    # Number of microphysics sub-steps inside one dynamics step.  Morrison's
    # double-moment product terms (q_c·q_r, q_i·q_c) run away at the
    # ~600s dynamics step; 10 sub-steps (60s each) keep it bounded.
    # Sundqvist / Kessler stay at 1 (no behaviour change).
    micro_substeps: int = 1
    # Aerosol-CCN coupling: diagnose the specified cloud-droplet number
    # from the prescribed aerosol optical depth (Andreae 2009 AOT–CCN
    # inversion) instead of the scheme's constant Nc_0.  Requires
    # aerosol_forcing="external" and a specified-Nc double-moment
    # microphysics (morrison); validated in build_physics_pipeline.
    nc_from_aerosol: bool = False
    # Sub-grid in-cloud warm-rain closure (#613): evaluate the non-linear
    # Morrison KK2000 autoconversion/accretion on in-cloud q_c (q_c / cloud
    # fraction), then scale back — recovers the drizzle that grid-mean rates
    # under-produce in partly-filled boxes at coarse resolution (raises precip
    # AND drains suspended cloud water -> lower LWP).  Morrison only; validated
    # in build_physics_pipeline.
    subgrid_autoconversion: bool = False

    # Sub-grid in-cloud autoconversion/accretion (Morrison & Gettelman 2008):
    # evaluate warm-rain rates on in-cloud q_c/cf and scale by cf so the
    # non-linear KK2000 rate is not under-fed by the grid-mean.  Requires
    # morrison microphysics.  Physics-fidelity correction (no tunable knob).
    subgrid_autoconversion: bool = False

    # Hard (iterated) saturation-adjustment guard, carried by EVERY guarded
    # scheme (microphysics/config.HARD_SAT_GUARD_SCHEMES: the five bulk
    # warm-rain ones plus sundqvist and ml_emulator): where q_v exceeds the
    # scheme's hard_sat_adjust_threshold * q_sat, an iterated saturation
    # adjustment drains q_v ONTO the liquid saturation curve (conserving
    # c_pd*T + L_v*q_v exactly), rate-limited to hard_sat_max_heating_K per step,
    # removing local super-saturation pools the smooth sigmoid path cannot.
    # PLACEMENT differs by dycore path (the integration trial showed placement is
    # load-bearing): on the SPECTRAL/COUPLED path it is applied IN-SCHEME
    # (threaded onto the per-scheme micro config via
    # apply_microphysics_experiment_flags); on the MPAS path it is applied
    # POST-STEP in model_driver._run_mpas, on the final state after the dycore's
    # vertical vapour transport -- the in-scheme placement cannot correct the
    # per-step transport spike within the dt window (it detonated at day 24),
    # while the post-step correction is the proven-stable intervention.  The
    # float trigger + heating cap default to the per-scheme __param_spec__
    # values (1.1, 5 K -- matching the validated configuration) and are
    # overridable via the flat scalars below (threaded through
    # apply_microphysics_experiment_flags on BOTH placement paths; routed for
    # --params via _ATM_SCALAR_PARAM_MAP -- the 2026-07-23 day-137 summer-
    # regime tuning need).  Default OFF => the moist path is byte-identical to
    # the smooth-only scheme.
    hard_saturation_adjustment: bool = False
    # Optional overrides of the hard-saturation-adjustment trigger + per-step
    # heating cap (None = per-scheme __param_spec__ defaults).  Setting either
    # without hard_saturation_adjustment=True is refused (silently-inert
    # configuration); bounds follow the scheme __param_spec__.
    hard_sat_adjust_threshold: float | None = None   # RH trigger, q_v > thr*q_sat
    hard_sat_max_heating_K: float | None = None      # per-step latent-heating cap [K]
    # Mixed-phase (ice-curve) drain: gate + land on the w(T)-blended
    # liquid/ice saturation curve with the blended latent heat below
    # freezing, and route cold condensate to cloud ice.  The liquid-only
    # drain leaves permanent ~60% ice-supersaturation at TTL temperatures —
    # the 20x-ERA5 TTL vapour bias (+17.8 K warm bias at 100 hPa) of the
    # first ClimateEval scorecard.  Requires hard_saturation_adjustment.
    hard_sat_ice_curve: bool = False
    # Homogeneous (Koop 2000 / Ren-MacKenzie 2005) cirrus ice nucleation in
    # Morrison.  The scheme's M2005 deposition needs PRE-EXISTING ice to
    # consume supersaturation, so with this off nothing caps RH_ice in
    # ice-free cirrus air: the century reached RH_ice = 288% at 228 K /
    # 222 hPa in mid-latitude storm tracks (2026-07-25 autopsy), which the
    # rate-limited drain then could not remove -> detonation.  Enabling it
    # adds the supersaturation-gated ice-NUMBER source that pins RH_ice near
    # S_hom ~ 1.45-1.6, i.e. treats the CAUSE rather than the symptom.
    # Morrison only; default OFF = byte-identical.
    homogeneous_ice_nucleation: bool = False

    # Convective in-updraft precipitation efficiency [0,1] (Tiedtke 1989 in-
    # updraft precipitation).  A value >0 diverts that fraction of the
    # convective condensate to rain (sediments via microphysics, invisible to
    # radiation) instead of detraining it all as suspended cloud.  Observed
    # deep-convective CPE ~0.5-0.9.  Supported by Tiedtke and Bechtold (threaded
    # in _resolve_convection).  SENTINEL: ``None`` (default) = use each scheme's
    # OWN default (Tiedtke 0.0 = legacy no-split; Bechtold 0.7 = ON, the #929
    # anvil-drain fix); an EXPLICIT value overrides it (0.0 forces the legacy
    # detrain-all path, dq_r None).  ``None`` distinguishes "unset" from an
    # explicit 0.0 so Bechtold's ON-by-default is not silently disabled.
    convective_precip_efficiency: float | None = None
    # Convective precip-split scheme (Bechtold / Tiedtke): "constant" uses the
    # fixed convective_precip_efficiency above; "autoconversion" derives the
    # precip fraction PHYSICALLY from the plume updraft cloud water (Sundqvist
    # 1978, convective_autoconversion_split), threaded to conv_config in
    # physics_pipeline; the scheme body raises on an unknown value.
    convective_precip_split: str = "constant"
    # True: the in-updraught convective rain surviving the scheme's own sub-cloud
    # evaporation leaves the column as surface precipitation (IFS); False: it is
    # handed to the microphysics rain tracer, which re-evaporated it at grid-mean
    # humidity (4.1 kg/m2/day below 700 hPa in the tropics).  DEFAULT True
    # (user 2026-09-16, with the cloud-cover threshold re-tuned on the drier
    # column): 30-day pair tropical-ocean prw -9 kg/m2, evaporation +4 %.
    convective_rain_to_surface: bool = True
    autoconv_q_c_crit: float = 5.0e-4   # [kg/kg] Sundqvist critical updraft cloud water
    autoconv_pe_max: float = 0.9        # [1] ceiling on the emergent precip fraction

    # Tiedtke plume buoyancy-death memory: when True the entraining plume,
    # once it exhausts its cumulative buoyancy budget, stays dead instead of
    # reviving above an inversion (the default False lets a plume killed by
    # negative buoyancy resume nonzero M_u aloft — physically questionable,
    # and the cause of convective detrainment heating reaching the ~100 hPa
    # tropical cold point).  Tiedtke-only (guarded in physics_pipeline).
    convective_buoyancy_death_memory: bool = False

    # Convection / Turbulence / GWD
    convection: str = "sbm"            # sbm, dca, kuo, mass_flux, edmf, zhang_mcfarlane, none
    turbulence: str = "none"           # smagorinsky, louis, tke, none
    # Surface-layer bulk-flux algorithm (SurfaceLayerConfig.bulk_scheme):
    # "constant" (neutral coefficients; DEFAULT, byte-identical) | "coare3" |
    # "large_yeager".  The constant scheme has NO convective-gustiness term, so
    # evaporation over a calm, convectively-unstable warm tropical ocean is
    # anemic (cold/dry surface-air bias).  The stability-dependent MOST schemes
    # add the free-convection velocity scale w*.  For interface energy
    # consistency the coupler ocean tile (CouplerConfig.bulk_scheme) MUST use the
    # same scheme — run_coupled wires both together.
    surface_bulk_scheme: str = "constant"
    # COARE 3.0 convective-gustiness BL depth z_i [m] for the MOST surface
    # fluxes (coare3/large_yeager): 0/None = off (byte-identical), ~600 = enable
    # the w* free-convection gust that lets a calm warm ocean evaporate
    # (the persistent tropical hfls<<Earth / R_TOA imbalance lever).  Threaded
    # into the atmosphere SurfaceLayerConfig + the slab SimpleOceanConfig.
    surface_gustiness_zi: float | None = None
    # Ocean surface-layer corrections (MPAS lane; both default off pending a
    # user decision): tell the MOST solver the lowest level's real height, and
    # use sea-water saturation at the surface pressure for the ocean q_sfc.
    # None = leave whatever the scheme config already carries (so a caller
    # who set it directly is not clobbered); True/False force it.  The AMIP
    # driver's own default is True -- see run_amip -- which is what makes the
    # correction the production default without this field silently
    # overruling an explicit scheme-level choice.
    surface_z_ref_model_level: bool | None = None
    # None = apply the sea-water surface humidity on every lane that can
    # honour it (today: MPAS, the only turbulence bridge carrying an ocean
    # fraction).  Explicit True on a lane that cannot is an error rather than
    # a silent no-op; explicit False reproduces the pre-2026-09-20 behaviour,
    # in which the atmosphere evaporated fresh water while the coupler's own
    # ocean tile evaporated sea water.
    surface_ocean_q_sfc_saline: bool | None = None
    # Thermodynamic constants set for the MOST surface fluxes (#762):
    # "legoesm" (default, byte-identical) = constant L_v / dry c_pd;
    # "aerobulk" = NEMO/AeroBulk/COARE parity (SST-dependent L_vap(T_sfc),
    # moist cp_air(q_atm)) — up to ~3 % LH at warm SST.  Threaded into the
    # atmosphere SurfaceLayerConfig (run_coupled additionally wires the slab
    # SimpleOceanConfig + coupler ocean tile to the same convention).
    surface_thermo_convention: str = "legoesm"
    # Stable-regime (zeta>0) MOST similarity functions for the MOST-family
    # surface bulk schemes.  Threaded into BOTH the atmosphere
    # SurfaceLayerConfig and the coupler ocean tile (CouplerConfig) by
    # run_coupled so the two sides of the interface always use the SAME
    # stable functions ("dyer1974" default = byte-identical: -5*zeta on the
    # constant/most/large_yeager Businger-Dyer path, and the COARE-native
    # stable form on coare3 — itself BH91 with rounded constants, so on
    # coare3 selecting "beljaars_holtslag1991" is a rounding-level change and
    # the genuinely different SBL tails are "grachev2007_sheba"/"gryanik2020";
    # see bulk_flux.psi_m_coare).
    surface_stability_scheme: str = "dyer1974"
    # Tiled (mosaic) surface fluxes: when True, the atmosphere surface
    # turbulent flux is computed SEPARATELY per surface tile and area-weighted
    # — ``surface_bulk_scheme`` (e.g. coare3) runs on the OCEAN tile, the
    # fixed-roughness land Monin-Obukhov scheme ("most", roughness
    # ``surface_z0_land``) runs on the LAND tile, and sea ice uses constant
    # coefficients — instead of applying one scheme to the blended surface
    # temperature (which runs the ocean air-sea scheme over land: the bm_v3
    # land-tile energy blowup).  Requires an active land tile + the louis
    # turbulence scheme (the only kernel that accepts the injected tiled flux).
    surface_tiled: bool = False
    surface_z0_land: float = 0.1       # land roughness length [m] for the tiled land MOST scheme
    # Prognostic soil-water bucket (Manabe 1969) for the slab-land tile: the
    # land evaporation efficiency beta = beta_min + (1-beta_min)*W/W_max
    # limits land latent heat by soil wetness (no longer a saturated swamp
    # everywhere).  Requires an active land tile.  ``w_land`` (soil water,
    # restart-persisted) advances from precip (source) and beta-limited land
    # evaporation (sink) with overflow runoff.  Same beta ramp as
    # legoesm.land.slab_land.  Off → byte-identical saturated-surface path.
    land_soil_bucket: bool = False
    land_bucket_w_max: float = 150.0      # soil-water bucket capacity [kg/m^2]
    land_beta_min: float = 0.1            # min moisture availability (dry soil)
    land_bucket_w_init_frac: float = 0.5  # initial soil water as fraction of W_max
    # Bucket runoff partition: Green-Ampt infiltration excess (Hortonian) +
    # saturation excess (Dunne).  Shared with legoesm.land.slab_land.
    land_K_infiltration: float = 1.0e-5     # saturated infiltration capacity K_s [m/s]
    land_infil_suction_boost: float = 2.0   # Green-Ampt suction enhancement psi_f/L_f [-]
    land_infiltration_excess: bool = True   # enable Hortonian infiltration-excess runoff
    # Route the soil-water availability through the SHARED land Jarvis (1976)
    # stomatal model (legoesm.land.stomata) instead of the bare bucket
    # ramp: beta = min(beta_soil, beta_canopy), the canopy term closing
    # stomata in low light / high VPD.  Requires land_soil_bucket (which
    # supplies beta_soil).  Off → soil-only bucket beta (byte-identical).
    land_stomatal_beta: bool = False
    # Global maximum stomatal (canopy) conductance [mol/m2/s] for the Jarvis /
    # Farquhar land stomata (StomataConfig.gs_max).  This is the calibration knob
    # for land evapotranspiration: gs = gs_max * f(PAR) * f(T) * f(VPD) * f(soil),
    # so lowering it raises canopy resistance and pulls land ET below potential.
    # Only Vc_max25 / g1 are PFT-overridden, so gs_max stays a clean *global*
    # lever (issue #730: the multilayer land over-transpires at potential because
    # the free-drainage equilibrium sits at field capacity where beta_root=1 with
    # no canopy resistance).  Default 0.3 matches StomataConfig.gs_max (byte-
    # identical when unchanged); only active when land_stomatal_beta=True.
    land_gs_max: float = 0.3
    # Multilayer-land surface scheme (#730). "simple_seb" (default) = bulk SEB
    # with the beta_soil moisture path; "two_leaf" = DifferBESS two-leaf canopy
    # energy balance (Kelvin h_r bare-soil evap + two-leaf stomatal transpiration),
    # which holds land ET below potential and breaks the over-evaporation wet loop
    # that the SimpleSEB beta_soil path (=1 at field capacity, no canopy resistance)
    # produces. Only affects use_multilayer_land runs.
    # DEFAULT: the two-leaf canopy.  It reads per-PFT CANOPY parameters, which
    # the driver now builds through the same routine the offline LMIP
    # simulations use; a canopy run without that surfdata is refused rather than
    # quietly given generic constants.
    #
    # "simple_seb" is ACADEMIC ONLY: measured on a well-watered column it
    # evaporates at potential with stomata off, and with them on its humidity
    # gradient SELF-EXTINGUISHES — the throttle warms the surface, saturation
    # rises, the throttle shrinks it back, and evaporation stops (2 W/m2 over
    # bare soil, surface 8-12 K hot).
    land_surface_scheme: str = "two_leaf"
    # Initial multilayer soil water as a fraction of saturation (theta_init =
    # frac * theta_sat) for the cold-start (#730). Default 0.5 is byte-identical to
    # the init_multilayer_land_state default. The multilayer over-evaporation wet
    # loop is precip-recycling-driven (land P ~= land ET), so a DRIER start (e.g.
    # 0.25) can tip the land into the slab-like dry attractor (less ET -> less low
    # cloud -> warmer land) instead of the cold-cloudy wet attractor. Only affects
    # use_multilayer_land runs.
    land_soil_moisture_init_frac: float = 0.5
    # HOW the multilayer soil is seeded at a cold start.
    #
    #   "aridity" (default, unchanged) — from the initial atmosphere's
    #       near-surface relative humidity, mapped into the plant-available
    #       range: theta = theta_wp + RH*(theta_fc - theta_wp).  Arid columns
    #       start near wilting, which is what stops a desert cold-start
    #       evaporation runaway (#730).  NOTE this caps the start at FIELD
    #       CAPACITY and makes ``land_soil_moisture_init_frac`` INERT — that
    #       field was only ever a fallback for an initial state carrying no
    #       humidity, which a real AMIP run never has (measured 2026-08-19: two
    #       arms differing only in the fraction were byte-identical).
    #
    #   "saturation_fraction" — theta = land_soil_moisture_init_frac * theta_sat,
    #       so the fraction becomes a real control and the soil can start ABOVE
    #       field capacity, up to near saturation.  Use it to keep a run out of
    #       the dry-soil attractor, where low soil water suppresses evaporation,
    #       which dries the boundary layer, which suppresses evaporation further.
    #       The trade is the runaway the aridity seed exists to prevent, so a
    #       wet start wants watching over the first week rather than trusting.
    land_soil_init: str = "aridity"
    # Prognostic snow + snow-albedo feedback on the AMIP slab-land tile: snow
    # water (SWE) accumulates from snowfall and melts (degree-day), brightening
    # the land albedo (snow ~0.5-0.8 vs vegetation ~0.15) — the positive
    # snow-albedo feedback SOTA AMIP land has.  Requires an active land tile.
    # Off (default) ⇒ static vegetation albedo (byte-identical legacy path).
    snow_albedo_feedback: bool = False
    gravity_wave_drag: str = "none"    # rayleigh, lindzen, mcfarlane, hines, prognostic_spectral, e3sm_cam, ml_emulator, none

    # Conservation
    fix_moisture: bool = False
    # Issue #323: make the per-step ``max(q_v, 0)`` floor on the physics
    # tracer update moist-static-energy-conserving (remove the latent heat
    # of the clipped vapour sink).  Opt-in for the kessler+sbm wind blow-up;
    # default off => bit-identical.
    energy_consistent_moisture_clip: bool = False
    # Resolved-wind moisture advection (issue #771): attach q_v/q_c/q_r/q_i/
    # q_s/q_g (+ the per-mass ice number N_i) to the dycore state each step so
    # the primitive-equation step advects them.  Without it, cube moisture is
    # COLUMN-LOCKED (physics tendencies + hyperdiffusion smoothing only) — the
    # wet-drift / day-150 blowup family.  Effective on cubed_sphere with the
    # cdgrid PE dycore (incl. the ``centered``/``finite_volume`` aliases that
    # resolve to cdgrid); other combos log a notice and keep the legacy path.
    #
    # OPT-IN / EXPERIMENTAL (default off => bit-identical): the transport is
    # ADVECTIVE form -(u·∇q), NOT flux form, so it does not discretely conserve
    # column/global water ∫ q·δp·dA under divergent flow — it drifts (pair with
    # ``fix_moisture_hydrostatic`` / ``energy_consistent_moisture_clip`` to
    # close the budget).  The per-VOLUME droplet/rain number densities N_c/N_r
    # are intentionally NOT advected here (a per-volume number is not a mass
    # mixing ratio; density-aware number transport is future work), so a
    # double-moment opt-in run advects the masses but leaves N_c/N_r
    # column-locked.  A flux-form mass-conserving tracer path is the follow-up.
    moisture_advection: bool = False

    # Topography
    topography: str = "flat"
    topo_smoothing: int = 4
    topo_edge_blend: float = 0.3
    # Optional land-sea-mask NetCDF (CMIP6 sftlf / ERA5 lsm).  When set,
    # the land fraction is taken from this file and the slab-land tile
    # is activated; empty → ocean-only surface.
    land_mask_path: str = ""
    # Optional land-surface-albedo NetCDF (e.g. ICON-extpar ALB on a
    # regular lat-lon grid).  When set AND ``land_mask_path`` is set, the
    # static land albedo field replaces the latitude-band fallback used
    # in ``ModelDriver._create_physics``.  ``albedo_land_month`` (1-12)
    # picks a single month from a monthly climatology; 0 → annual mean.
    albedo_land_path: str = ""
    albedo_land_month: int = 0
    # Optional subgrid orographic stddev NetCDF (ICON-extpar ``SSO_STDH`` on a
    # regular lat-lon grid, e.g. ``extpar_sso_latlon_0p25deg.nc``).  When set
    # AND an orographic GWD scheme (``mcfarlane`` / ``lindzen``) is active, the
    # per-column launch height ``h_topo`` is taken from this field (real
    # mountains, ~0 over ocean) instead of the scalar ``McFarlaneConfig.h_topo``
    # = 500 m everywhere.  Empty → scalar fallback (legacy behaviour).
    subgrid_orography_path: str = ""
    # Slab-land surface-energy-balance knobs.  Only relevant when the
    # land tile is active (``land_mask_path`` set).
    #   C_land           : effective heat capacity [J/m^2/K]
    #   emissivity_land  : LW emissivity
    #   beta_land        : soil-moisture evaporation factor in [0, 1]
    #                      (1 = wet surface; calibration default in
    #                      Phase 1 — left tunable)
    C_land: float = 2.0e5
    emissivity_land: float = constants.emissivity_land
    beta_land: float = 1.0

    # Multilayer land surface (Phase L1).  When True, replaces the slab
    # _step_slab_land call with step_multilayer_land from legoesm.land,
    # which carries a prognostic (T_soil, theta_soil, psi_soil, snow)
    # state per column.  Requires land_mask_path to also be set.  The
    # slab knobs (C_land, emissivity_land, beta_land) become unused and
    # the multilayer config below takes over.
    use_multilayer_land: bool = False
    # Land-tile call interval [s]; 0.0 (default) advances the land every
    # host step (legacy).  A positive value calls the tile every
    # round(land_update_seconds/dt) steps with the atmosphere forcing
    # AVERAGED over the interval (mass/energy of the interval conserved)
    # and the tile's fluxes/skin held in between — the same shape as the
    # hourly radiation cadence.  MPAS multilayer-land lane only.
    land_update_seconds: float = 0.0
    multilayer_n_layers: int = 10        # soil discretization
    multilayer_soil_depth: float = 3.0   # m
    # Run the multilayer land tile in EXACTLY the configuration its baked
    # per-PFT tables were calibrated under (the single definition lives in
    # ``legoesm.land.config.calibrated_multilayer_setup``): MOST surface
    # exchange, the SimpleSEB surface scheme, Farquhar stomata on a PRESCRIBED
    # carbon state, and the calibration soil column (8 layers, 3 m, geometric
    # growth 1.5).  Off (default) leaves every existing run's numerics
    # unchanged.
    #
    # Without it the baked canopy conductance (Vc_max25/g1/LCMA) is INERT: the
    # coupled tile took the no-stomata branch of ``compute_effective_beta``, so
    # the tables were deployed under land physics they were never fitted to.
    # Setting ``land_stomatal_beta`` alone does NOT fix that — with no carbon
    # state the same dispatch falls to JARVIS, a different stomatal model.
    # Validation below requires the overlapping config keys to already carry the
    # calibration values, so nothing is silently overridden.
    land_calibrated_physics: bool = False
    # CLM-ML only: derive each column's PFT from the surface map's DOMINANT PFT
    # (argmax of pft_fractions) instead of one pft_clm for all columns -> mixed-PFT
    # heterogeneous columns, compiled at O(#distinct structures) by the group-by-
    # structure canopy scan.  Opt-in (default False = single pft_clm, no change): the
    # dominant-PFT map moves bare/other columns off pft_clm, so a faithful run wants
    # the full CLM PFT parameterisation validated.  Ignored unless land_surface_scheme
    # == 'clm_ml' (+ use_multilayer_land).
    clm_ml_use_surfdata_pft: bool = False
    # Strategy B: hydrate MultiLayerLandState from an ERA5 land NetCDF
    # at IC time instead of the strategy-A uniform 0.5*theta_sat fill.
    # Build with probes/preprocess_era5_land_ic.sh; expects variables
    # stl1-4, swvl1-4, sd on a regular lat-lon grid.  Ignored when
    # use_multilayer_land is False.
    era5_land_ic_path: str = ""
    # Spun-up land INITIAL CONDITION (#746 item 1): a MultiLayerLandState
    # restart (.npz) written by ``scripts/run/run_land_spinup.py`` after an
    # offline multi-year land spin-up.  When set (and use_multilayer_land is
    # True) it REPLACES the cold-start ``init_multilayer_land_state`` +
    # aridity-theta seed with the equilibrated soil column, so a coupled AMIP
    # run starts from a settled deep-soil temperature/moisture instead of the
    # day-0 cold-start shock that drives the land cloud-albedo cold trap.  The
    # restart's ncol / n_layers must match the run's grid (validated on load).
    # Takes precedence over era5_land_ic_path when both are set.
    land_ic_path: str = ""
    # Pre-staged CLM surfdata NetCDF (PFT/texture/glacier maps) for the multilayer
    # land.  Empty => download from UCAR to /tmp (fails on compute nodes with no
    # outbound internet, so stage the file and set this).  The multilayer soil
    # column reads it only when use_multilayer_land is True; ADDITIONALLY (and
    # regardless of use_multilayer_land) a staged path supplies RADIATION's
    # static land albedo via the ERA5-tuned CLM map when neither
    # albedo_land_path nor surfdata_path is set — a slab/no-land-model AMIP run
    # staging this file deliberately gets the calibrated albedo instead of the
    # latitude fallback (2026-08-10; previously the fallback silently won).
    clm_surfdata_path: str = ""

    # Transient land-use/land-cover (LULC).  ``land_cover_surfdata`` is a harmonized
    # transient legoesm_surfdata NetCDF (``pft_frac(year, npft, lat, lon)`` in
    # percent on the CLM5 17-PFT axis; built by ``scripts/data/build_*_surfdata.py``
    # from LUH2/HYDE/Pongratz/KK10).  When ``transient_land_cover`` is True (and
    # use_multilayer_land is True) the per-column multilayer VEGETATION params
    # (albedo / z0 / root / emissivity / stomata + the plant btran wilting/field-
    # capacity thresholds) are re-weighted every segment at
    # ``cover_year = start_year + elapsed_days/365`` (``interp_annual`` on the annual
    # cover); per-cell SOIL texture/hydraulics/thermal + prescribed LAI stay frozen
    # (land use changes vegetation, not soil; transient LAI is a documented
    # follow-up).  The rebuilt ``LandSurfaceParams`` is passed to the jitted step as
    # a TRACED per-segment ``SegmentForcing`` arg (not the closure-baked
    # ``pipeline.land_ml_params``), so the compiled AMIP step follows the evolving
    # cover with NO retrace.  Off (default) => static single-year cover, byte-
    # identical to the pre-transient behaviour.  Ignored unless use_multilayer_land.
    transient_land_cover: bool = False
    land_cover_surfdata: str = ""

    # Diagnostic T-based ice partition.  At every radiation call the
    #   cloud_r_eff_ice : ice effective radius [m]
    cloud_r_eff_ice: float = 30.0e-6

    # Land/ocean cloud droplet effective radius (Phase 3 of cloud-micro plan).
    # Land has ~3x higher CDNC than ocean, giving ~30% smaller r_eff and
    # brighter clouds.  Compute the per-column r_eff as a linear blend
    # weighted by f_land (the land-fraction field) inside
    # compute_radiation_core.  Replaces cloud_r_eff_liq in the AMIP active
    # tunable set; the single-value field is kept for backward-compat with
    # no-land runs (f_land = 0 → blend equals cloud_r_eff_liq_ocean).
    cloud_r_eff_liq_ocean: float = 10.0e-6
    cloud_r_eff_liq_land: float = 7.0e-6
    # Activate the slab-land SEB tile without loading a separate LSM file.
    # Useful when --topography already provides a good f_land (ETOPO) and
    # no separate mask file is available.  Ignored when land_mask_path is set
    # (the file path already implies activation).
    slab_land_active: bool = False
    # Optional harmonized surface-data NetCDF (legoesm_surfdata_*.nc).  When
    # set together with land_mask_path, the static land albedo field is taken
    # from the surfdata (per-column soil-colour + PFT-vegetation blend, glacier
    # override) instead of the latitude-only land_vegetation_albedo() curve;
    # empty → latitude-only land albedo (unchanged behaviour).
    surfdata_path: str = ""

    # Surface
    T_init: float = 300.0
    rh_init: float = 0.7
    dynamic_albedo: bool = False
    carbon_cycle: str = "none"

    # Initial conditions
    #   "default"  — isothermal held_suarez_init at T_init (e.g. 300 K)
    #   "standard" — realistic lapse-rate profile (held_suarez_init with the
    #                temperature replaced by a standard atmosphere)
    #   "era5"     — ERA5 reanalysis snapshot (requires ic_path)
    ic: str = "default"
    ic_path: str = ""     # ERA5 Zarr path when ic="era5"

    # CMIP
    experiment: str = ""
    start_year: int = 1979

    # Surface parameters
    C_H: float = 0.0044
    C_E: float = 0.0044
    # T_ice is the seawater freezing point used as the SST floor /
    # SIC ramp threshold — NOT the ice surface temperature.  Legacy
    # name kept for AMIP config compatibility.
    T_ice: float = constants.T_freeze_ocean
    # Flat over the prescribed ice fraction (no snow-on-ice / pond / temperature
    # dependence on the MPAS lane). 0.80 = snow-covered late-winter ice
    # (CICE/CCSM3 dry snow-on-ice broadband); 0.65 was a bare/melting value.
    # DEFAULT moved 0.65 -> 0.80 (user 2026-09-15), see land_snow_tau_days.
    albedo_ice: float = 0.80
    albedo_ocean: float = 0.06
    sfc_emissivity: float = constants.emissivity_ocean
    emissivity_ice: float = constants.emissivity_ice
    tau_equator: float = 7.2
    tau_pole: float = 1.8
    tau_moist_coeff: float = 0.0115        # gray-rad moisture LW optical depth [m²/kg]
    # Additional gray-radiation knobs (GrayRadiationConfig fields), exposed on
    # ExperimentConfig so the calibration can tune them.  They are threaded to
    # the gray radiation kernel as a `gray_cfg_overrides` dict.  NOTE:
    # sfc_emissivity (above) is NOT honoured by gray radiation — gray keeps its
    # idealized black-surface convention (eps=1.0, Held-Suarez/Frierson); the
    # sfc_emissivity value feeds RRTMGP and the dynamic surface-emissivity blend
    # instead (see the gray builder's `del emis_col` note in physics_pipeline).
    linear_frac: float = 0.2               # linear vs sigma^4 LW weighting
    lw_diff_factor: float = 1.66           # LW diffusivity factor D
    sw_tau_0: float = 0.22                 # SW optical-depth scale
    sw_exponent: float = 2.0               # SW optical-depth vertical exponent
    sbm_tau_c: float = 7200.0
    sbm_RH_ref: float = 0.7
    sbm_cape_threshold: float = 70.0
    sundqvist_auto_rate: float = 1e-3      # Sundqvist autoconversion rate [1/s]
    sundqvist_evap_coeff: float = 5e-4     # Sundqvist sub-cloud rain evaporation coeff
    cloud_r_eff_liq: float = 10.0e-6       # cloud droplet effective radius [m]
    # Convection / microphysics / turbulence / GWD scheme knobs exposed for
    # calibration (AIMIP commit 0c747d4).  These mirror the scheme-config
    # defaults; the calibration threads them to the schemes as a
    # `physics_cfg_overrides` dict-of-dicts (keys: micro/conv/turb/gwd).
    sundqvist_sigmoid_sharpness: float = 20.0   # SundqvistConfig.sigmoid_sharpness
    sbm_T_min_convect: float = 200.0            # SBMConfig.T_min_convect [K]
    louis_l_mix_max: float = 100.0              # LouisConfig.l_mix_max [m]
    # Marine-Sc cloud-top entrainment (Louis BL): vents trapped BL-top moisture
    # into the dry free troposphere to thin excess stratocumulus liquid cloud
    # (the AMIP albedo bias) without a surface-evaporation trade.  SINGLE knob:
    # 0.0 = off (default => byte-identical), > 0 = on.  Deploy warm-start/ramp.
    louis_cloudtop_entrainment_efficiency: float = 0.0  # LouisConfig.cloudtop_entrainment_efficiency [0,1]; 0=off
    louis_Ck: float = 0.4                       # LouisConfig.Ck
    louis_Ri_crit: float = 0.25                 # LouisConfig.Ri_crit
    louis_b_louis: float = 5.0                  # LouisConfig.b_louis
    louis_c_louis: float = 16.6                 # LouisConfig.c_louis
    louis_d_louis: float = 5.0                  # LouisConfig.d_louis
    louis_z0: float = 1.0e-4                    # SurfaceLayerConfig.z0 [m]
    louis_Ch_neutral: float = 1.5e-3            # SurfaceLayerConfig.Ch_neutral
    louis_Cd_neutral: float = 1.5e-3            # SurfaceLayerConfig.Cd_neutral
    # Exact 2*pi/100 km — MUST equal McFarlaneConfig.k_wave's own
    # expression: gwd_config_for overlays this onto the leaf, so a
    # truncated literal would silently perturb the default kernel.
    mcfarlane_k_wave: float = 2.0 * math.pi / 100e3  # McFarlaneConfig.k_wave [1/m]
    # INERT: no McFarlaneConfig field of this name exists — the scheme derives
    # N from the column state (mcfarlane.py).  Kept only for the positional ABI
    # + serialized-config compatibility; gwd_config_for deliberately does not
    # wire it, and it was dropped from the ml/tuning.py catalog so it can no
    # longer be advertised as a live knob (codex round 1, finding 5).
    mcfarlane_N_ref: float = 0.01               # INERT (no leaf field)
    mcfarlane_directional_spread: float = 1.0   # McFarlaneConfig.directional_spread
    mcfarlane_tau_max: float = 10.0             # McFarlaneConfig.tau_max [Pa]
    # Morrison ice-microphysics tunables (active when microphysics='morrison'
    # — silently filtered out by hasattr when Sundqvist/Kessler is active).
    morrison_bergeron_rate: float = 1e-3        # MorrisonConfig.bergeron_rate [1/s]
    morrison_rime_coeff: float = 1.0            # MorrisonConfig.rime_coeff
    morrison_dep_coeff: float = 1e-3            # MorrisonConfig.dep_coeff
                                                # (was 1e-8 — 1e5 OFF the real
                                                # leaf default while unwired;
                                                # flag-reachability audit
                                                # 2026-07-25.  MUST equal the
                                                # MorrisonConfig default so
                                                # wiring is a no-op at rest.)
                                                 # (Morrison-2005 q_i^(1/3)·N_i^(2/3) form)
    morrison_agg_coeff: float = 1e-3            # MorrisonConfig.agg_coeff [1/s]
    morrison_k_au: float = 6e2                  # MorrisonConfig.k_au [1/(kg*s)]
    # Anvil-ice sink/source scalars (2026-07-27 warm-drift diagnosis: IWP 34x
    # obs because small-crystal cirrus defeats both ice sinks while hom
    # nucleation keeps seeding number — these three knobs bracket the loop).
    morrison_fall_a_i: float = 700.0            # MorrisonConfig.fall_a_i [m^(1-b)/s]
    morrison_ice_snow_d_auto: float = 250.0e-6  # MorrisonConfig.ice_snow_d_auto [m]
    morrison_hom_ice_nuc_N: float = 1.0e6       # MorrisonConfig.hom_ice_nuc_N [1/m^3]
                                                # (consumed only inside the
                                                # homogeneous_ice_nucleation
                                                # branch; validate_strict
                                                # refuses the inert combo)
    # SBM convective precip efficiency: fraction of column-net drying that
    # precipitates directly as rain (rest is detrained as condensate).
    sbm_precip_efficiency: float = 0.5          # SBMConfig.precip_efficiency
    # (morrison_phase_aware_sat_adj / cloud_rh_ice_crit / cloud_rh_ice_sat
    # DELETED 2026-07-26: their comments documented physics that was NEVER
    # implemented — no MorrisonConfig.phase_aware_sat_adj, no
    # CloudConfig.rh_ice_crit/rh_ice_sat exist anywhere (flag-reachability
    # audit cause 3, codex-verified).  Old serialized configs carrying them
    # load fine: the known-field filter drops unknown keys.)
    # Bechtold deep-convection CAPE trigger threshold [J/kg].  Deep convection
    # fires only above this CAPE; lowering it lets convection trigger more
    # readily at coarse resolution (where CAPE is under-resolved), which is
    # the lever for the AMIP convective-precipitation deficit.  Default matches
    # BechtoldConfig.cape_threshold (byte-identical when unset).
    bechtold_cape_threshold: float = 70.0
    # Bechtold compensating-subsidence vertical solve (BechtoldConfig.
    # subsidence_solve): "implicit_flux" (conservative, the 2026-07-22
    # default) or "advective" (legacy, truncation-order conservation only,
    # kept selectable for byte-exact reproduction and as a stability
    # escape hatch while the implicit bottom-boundary behaviour is under
    # investigation — day-65 pilot blowup bisect, 2026-07-22).
    bechtold_subsidence_solve: str = "implicit_flux"
    # Where the in-plume rain's vapour is debited (BechtoldConfig.rain_vapor_sink):
    # "formation" (default, at the rain-formation levels) or "vapour_mass"
    # (legacy spread over the whole column by vapour mass; the A/B control).
    bechtold_rain_vapor_sink: str = "formation"
    # Bechtold convective-top pressure [Pa]; terminates the (non-detraining)
    # plume + subsidence gate. 150 hPa stability cap (see BechtoldConfig.
    # p_conv_top_pa); raise toward 100 hPa if deep tropical tops are clipped.
    bechtold_conv_top_pa: float = 15000.0
    # Bechtold convective-downdraft strength (marine-evaporation / precip lever,
    # #847). The downdraft's sub-cloud effect is rain re-evaporation only — it
    # COOLS + locally moistens (no dry-air advection; see physics_pipeline note).
    # Raising downdraft_evap (0.05 -> ~0.3 Tiedtke) increases that cooling, which
    # drives cold pools that ENHANCE convective triggering -> more precip -> net
    # column drying -> larger sea-air gradient -> higher surface evaporation.
    # downdraft_rh_min gates the trigger sigmoid((rh_min - rh_below)·sharpness):
    # the downdraft fires where the below-LCL RH < rh_min, so RAISING rh_min
    # activates it in more (moister) columns. Defaults reproduce BechtoldConfig
    # (byte-identical).
    bechtold_downdraft_evap: float = 0.05
    bechtold_downdraft_alpha: float = 0.3
    bechtold_downdraft_rh_min: float = 0.2
    # Penetrative-downdraft thermodynamic transport (marine-BL ventilation).
    # The re-evaporation downdraft above only MOISTENS the sub-cloud layer;
    # transport ON advects low-MSE (dry) mid-level air DOWN into the BL,
    # DRYING it -> stronger surface evaporation + less BL liquid cloud (lower
    # albedo).  Default OFF => byte-identical to the re-evaporation-only
    # downdraft.  See BechtoldConfig.downdraft_transport.
    bechtold_downdraft_transport: bool = False
    bechtold_downdraft_entrain_rate: float = 3.0e-4  # IFS ENTRDD (sucumf.F90:144)
    bechtold_downdraft_detrain_scale_m: float = 700.0
    # Full IFS deep CAPE closure ZMFUB1=ZCAPE*ZMFUB/(ZHEAT*ZXTAU)
    # (openifs cumastrn.F90:704-833; PR #1095).  Default ON (2026-07-16,
    # validated: codex x11 + gray-RCE A/B + C24 AMIP smoke A/B, both stable /
    # neutral); --no-bechtold-use-ifs-cape-closure restores the legacy
    # surrogate byte-identically.  Mirrors BechtoldConfig.use_ifs_cape_closure.
    bechtold_use_ifs_cape_closure: bool = True
    # IFS Kessler sub-cloud rain evaporation (cuflxn.F90:436-475).  Default
    # ON (2026-07-16, validated: codex x3 + gray-RCE A/B + C24 AMIP A/B);
    # mirrors BechtoldConfig.use_ifs_subcloud_evap (fallbacks match).
    bechtold_use_ifs_subcloud_evap: bool = True
    # IFS in-updraft precipitation formation (cuascn.F90:718-773).  Default
    # ON (2026-07-16, validated: codex x3 + gray-RCE A/B + C24 AMIP A/B);
    # mirrors BechtoldConfig.use_ifs_inplume_precip (fallbacks match).
    bechtold_use_ifs_inplume_precip: bool = True
    # Grid spacing [m] for the IFS ZTAURES turnover resolution factor
    # (cumastrn.F90:762-768).  0 = legacy factor 1.0.  Mirrors
    # BechtoldConfig.dx_m.
    # IFS in-plume conversion constants (anvil-source control, 2026-07-27):
    # more conversion (rprcon up / dnoprc down) = drier detrained outflow.
    bechtold_rprcon: float = 1.4e-3   # BechtoldConfig.rprcon [1/m]
    # Cloud-base mass-flux cap [kg/m^2/s]. The pipeline used to read a field of
    # this name that did not exist and fell back to 0.02 (2.5x tighter than the
    # scheme's own 0.05), which bound in 71 % of tropical-ocean columns.
    # DEFAULT 0.05 = the scheme's own value (user 2026-09-16); 5-day arms at
    # 0.05/0.10 were stable with small effects (cover -2/-5 pts).
    bechtold_M_b_max: float = 0.05   # BechtoldConfig.M_b_max [kg/m^2/s]
    # Gregory-1997 convective momentum transport, applied to MPAS edge winds
    # through the shared cell->edge projection (2026-09-15).  None preserves
    # each lane's earlier behaviour: OFF on MPAS (where the bridge handed the
    # scheme zero winds, so CMT was inert) and the scheme's own default (ON)
    # elsewhere.  The MPAS production default is a pending user decision.
    bechtold_enable_cmt: bool | None = None
    bechtold_dnoprc: float = 3.0e-4   # BechtoldConfig.dnoprc [kg/kg]
    # Deep-plume entrainment / detrainment base rates (IFS cuascn), exposed
    # 2026-08-14. These set the ITCZ WIDTH and tropical rain concentration:
    # raising epsilon_deep dilutes the deep plume faster in dry air, so
    # convection survives only where the column is already moist -> a narrower,
    # wetter rain band. Both are scaled in-scheme by the IFS height/RH factors;
    # these are the BASE rates. Defaults are the published IFS deep values and
    # are byte-identical to the previous hard-coded behaviour.
    bechtold_epsilon_deep: float = 1.75e-3   # BechtoldConfig.epsilon_deep [1/m]
    bechtold_delta_deep: float = 0.75e-4     # BechtoldConfig.delta_deep [1/m]
    bechtold_dx_m: float = 0.0
    # IFS convective downdraft (cudlfsn+cuddrafn).  Default ON since
    # 2026-07-17 (RCE/AMIP A/B); mirrors BechtoldConfig.use_ifs_downdraft.
    bechtold_use_ifs_downdraft: bool = True  # flipped 2026-07-17 (RCE/AMIP A/B)
    # IFS shallow PBL-equilibrium closure (cumastrn.F90).  Default STILL
    # OFF — HELD by the 2026-07-17 flip campaign (largest mean-state
    # reshape; needs a skill-gated run); mirrors
    # BechtoldConfig.use_ifs_shallow_closure.
    bechtold_use_ifs_shallow_closure: bool = False
    # IFS diurnal CAPE correction + land RH break (default ON since
    # 2026-07-17); mirror BechtoldConfig.use_ifs_capdcycl / use_ifs_land_rhebc.
    bechtold_use_ifs_capdcycl: bool = True
    bechtold_use_ifs_land_rhebc: bool = True
    bechtold_use_ifs_snow_melt: bool = True
    sigma_b: float = 0.7
    k_BL_max_per_day: float = 1.0
    k_free_per_day: float = 0.1
    # Top-of-atmosphere sponge (#836): a Rayleigh damping increasing toward the
    # model lid to absorb upward-propagating gravity-/convective-wave energy.
    # The k_BL drag above is maximal at the SURFACE, so the hydrostatic
    # latlon-cgrid dycore otherwise has NO top sponge -> waves reflect off the
    # rigid ~35 hPa lid (upper-level noise; blocks aggressive cloud-thinning
    # calibration).  OFF by default (byte-identical); enabled in the reference
    # AMIP config.  sin^2 ramp from 0 at sigma=sponge_sigma_top to
    # sponge_coeff_per_day at the model top; folded into the existing fric_decay.
    sponge_enabled: bool = False
    sponge_coeff_per_day: float = 2.0   # Rayleigh damping rate at the model top [1/day]
    sponge_sigma_top: float = 0.15      # sponge base: sigma below which damping ramps up

    # --- MPAS land surface boundary (standalone MPAS lane only) -------------
    # The MPAS combined-physics lane has NO land tile: the AMIP loader fills
    # land cells with the NEAREST-OCEAN SST (a sea-level temperature) and the
    # surface humidity is saturated everywhere, so every land cell acts as a
    # warm infinite swamp (+5-12 K vs its own air over elevated terrain).
    # Instrumented on the 2026-07-23 pilot: tropical-land latent flux 110-240
    # W/m2 (vs 74 ocean), SBM precip up to 19 mm/day on highlands, and a
    # cell-scale CWV recharge/discharge speckle.  Two first-order corrections:
    # lapse-adjust the land anchor (T_eff = T_sfc - f_land*lapse*z) and
    # throttle the land evaporation efficiency (beta).  Defaults are OFF /
    # byte-identical.  FV / spectral lanes have a real land tile — these
    # knobs are refused there (validate_strict).
    mpas_land_lapse_K_per_km: float = 0.0  # land anchor lapse [K/km]; 0=off, 6.5=ICAO std
    mpas_land_beta: float = 1.0            # land evaporation efficiency [0-1]; 1=wet swamp
    # Phase 2b (#1312): traced per-cell root-zone beta_soil from the
    # interactive multilayer land -> the MPAS turbulence surface humidity
    # (forcing["beta_land"], one-step lag like the skin-T blend).  Replaces
    # the STATIC mpas_land_beta over land when on (the land latent flux is
    # then throttled by the soil's own moisture state — the same
    # land_tile_beta_soil the coupled pipeline applies).  Requires
    # use_multilayer_land on the MPAS lane; default OFF = byte-identical.
    # NOTE: this now publishes the land tile's SOLVED surface humidity AND its
    # solved sensible/latent fluxes to the atmosphere, not merely a root-zone
    # beta — the flux handoff replaced the humidity-only one after the latter was
    # measured to deliver about a tenth of the solved flux.  With it off the mesh
    # lane discards all three and keeps the static ``mpas_land_beta``.
    mpas_land_beta_soil: bool = False

    # Held-Suarez forcing
    held_suarez_forcing: bool = False  # add HS Newtonian relaxation + Rayleigh drag

    # Joint ML physics parameterization
    physics_parameterization: str = "none"  # none, ml
    physics_parameterization_checkpoint: str = ""
    physics_parameterization_stats: str = ""
    physics_parameterization_hidden_dim: int = 128
    physics_parameterization_layers: int = 3
    physics_parameterization_seed: int = 0

    # AIMIP intercomparison variant tag.  Empty string => not an AIMIP run
    # (preserves backward compatibility for all existing AMIP configs).
    # When set, ``scripts/run/run_aimip.py`` dispatches to the matching
    # training entry point and ``validate_strict`` enforces the
    # corresponding scheme prerequisites.
    aimip_variant: str = ""  # "", classical, column_nn, sfno_physics, sfno_full

    # Performance
    precision: str = "fp32"           # fp32, fp64, mixed, or mixed_fp64_storage
    gradient_checkpoint: bool = False  # wrap scan body with jax.checkpoint for AD
    debug_precision: bool = False     # log warnings when array dtypes mismatch policy

    # Reproducibility (Stage A1).  Master RNG seed for the run: every random key
    # descends from this via ``legoesm.runtime.rng.split_keys``, so the run is
    # reproducible from the seed recorded in the run manifest.
    seed: int = 0

    # Segment/forcing cadence when NO host cadence exists (diag_days=0 AND
    # checkpoint_days=0 — e.g. distributed_mode='spmd' milestone-1): the
    # compiled-segment fallback length, which is ALSO how often
    # time-varying forcing (SST/SIC, solar, ozone/aerosol, coupler
    # overrides) is re-sampled — forcing updates only at segment
    # boundaries.  Ignored whenever diagnostics or checkpoints set a
    # finer cadence.  Without a fallback the segment collapses to 1 step
    # and every step pays a host boundary (multi-controller: a
    # cross-process rendezvous per step — the production-SPMD
    # anti-scaling, job 8471423).
    forcing_update_days: float = 1.0

    # Distributed
    distributed: bool = False
    # How multi-process runs federate (read only when distributed=True):
    #   "mpi"  — mpi4jax halo backend: replicated cubed-sphere dynamics
    #            (full 6-face state per rank, physics-only scatter),
    #            lat-lon band, or Voronoi cell partition.  Legacy default.
    #   "spmd" — multi-controller jax.distributed: ONE global device
    #            mesh, true cubed-sphere domain decomposition (the bench
    #            --cs-spmd path productionised; shard-local parity
    #            6.7e-10 @5 steps, job 8462928).  Cubed-sphere only;
    #            mpi4jax is NEVER armed in this mode — mpi4jax and
    #            jax.distributed collectives in one program is the
    #            documented mixed-stack deadlock.
    distributed_mode: str = "mpi"
    ensemble_size: int = 1
    n_devices: int | str = "auto"  # number of GPUs, or "auto" for all visible
    # Issue #273 follow-up: opt-in horizontal-column sharding for the
    # per-column radiation kernel.  Decouples per-column physics
    # throughput from cubed-sphere face-divisibility (4-GPU node
    # unblock).  Requires ``6 · n · n`` (the flattened column count)
    # divisible by the active device count — typically holds for
    # production resolutions (C16=1536, C48=13824).  Default off
    # preserves bit-exact behavior.
    shard_radiation_columns: bool = False
    # Issue #273 follow-up: opt-in level-parallel cubed-sphere mesh
    # for device counts that fail face-sharding divisibility (e.g.
    # 4 on a 4×A100 node).  When True, ``bootstrap()`` routes the
    # dycore mesh to ``cubed_sphere_level`` (replicated horizontal
    # stencil, level-sharded) instead of clamping to the nearest
    # face-compatible count.  Pair with ``shard_radiation_columns``
    # for the full 4-GPU unblock.  Default off.
    allow_level_fallback: bool = False
    # A1 (lat-lon SPMD): opt-in single-process multi-device lat-BAND
    # decomposition for the lat-lon C-grid hydrostatic dycore (the atm twin
    # of the ocean lat-band SPMD step).  Routes ModelDriver.run() to the
    # dedicated ``_run_compiled_latlon_spmd`` segment loop
    # (``run_atm_latlon_spmd``).  DISTINCT from ``distributed_mode='spmd'``
    # (that is the multi-controller cubed-sphere path); this is the
    # single-process ``n_devices>1`` path and never arms mpi4jax.  Requires
    # grid.grid_type='latlon', n_lat % n_devices == 0, and DYNAMICS-ONLY or a
    # STATELESS physics (Held-Suarez / per-column); a stateful PhysicsState
    # carry is not yet SPMD-routed.  Default off preserves all existing paths.
    enable_latlon_spmd: bool = False
    # P4 (cube >6 devices): opt-in sub-face-TILED dynamics — the compiled
    # segment's dynamics core routes through
    # ``make_tiled_cc_step`` (tiled D-grid SSP-RK3 on a (6,kt,kt) mesh)
    # when the device layout is sub-face tiled (n_devices = 6*kt^2 > 6).
    # Dynamics-only swap: physics/fixers/tracers in the segment are
    # untouched.  The adapter refuses configs outside the tiled base-cut
    # envelope (non-ssp_rk3 integrator, any extra damping term, duogrid)
    # LOUDLY.  Default off preserves every existing path.
    enable_tiled_dycore: bool = False
    # M2b (scaling): run each lat-lon SPMD segment as ONE compiled
    # ``lax.scan`` (``make_sharded_atm_latlon_segment`` — band-sharded
    # geometry, one host dispatch + one in-graph finite-scalar read per
    # segment) instead of the historical per-step Python loop.  Applies to
    # the STATELESS ``run_atm_latlon_spmd`` lane (dynamics-only /
    # Held-Suarez); the operator-split unified-physics SPMD lane has no
    # compiled-scan segments yet and REFUSES this flag loudly (never a
    # silent no-op).  Requires ``enable_latlon_spmd=True`` (validated).
    # Default off = byte-identical per-step path.
    latlon_spmd_compiled_segments: bool = False

    # Optional explicit turbulence scheme config (a
    # ``atmosphere.physics.turbulence.config.TurbulenceConfig``) overriding the
    # default ``TurbulenceConfig(scheme=turbulence)`` that the driver builds from
    # the scheme STRING.  Lets a caller inject a refined / per-column scheme
    # sub-config (e.g. a corrected per-column ``clubb_lite.C_K`` field from the
    # LES-informed correction loop) WITHOUT a new driver signature.  Its
    # ``.scheme`` MUST equal ``turbulence`` (it refines the same scheme, it does
    # NOT switch schemes — validated in ``validate_strict``).  ``None`` (default)
    # ⇒ the driver builds the default config, byte-identical to before.  Typed
    # ``Any`` to avoid importing the atmosphere physics config into the driver
    # config module.
    turbulence_override: Any = None
    # Full ``GravityWaveDragConfig`` override, mirroring ``turbulence_override``:
    # ``.scheme`` MUST equal ``gravity_wave_drag`` (refines the same scheme —
    # validated in ``validate_strict``).  ``None`` (default) ⇒ the driver builds
    # ``GravityWaveDragConfig(scheme=...)``, byte-identical to before.  This is
    # the ONLY coupled-path route to nested GWD scheme options
    # (``mcfarlane.use_e3sm_hdsp``, ``e3sm_cam.use_discrete_ke_heating``, tuned
    # ``fcrit2``, ...); without it they were silently discarded.
    gravity_wave_drag_override: Any = None

    # Horizontal q_v smoothing on the MPAS lane [m^2/s]; 0 = off (byte-
    # identical; appended at the tuple END to preserve the positional ABI).
    # The MPAS lane historically had NO horizontal moisture smoothing (the FV
    # lanes smooth q_v every step in their step factories) — the confirmed
    # missing third suspect behind the cell-scale CWV recharge/discharge
    # speckle (2026-07-23).  Applied post-step as an UNWEIGHTED SCVT del2 (∇²)
    # + a q>=0 floor: monotone / positivity-preserving under the setup-time
    # CFL guard (the floor is then a no-op, to roundoff), and it conserves the per-level
    # mixing-ratio area integral sum_c A_c q_c (exact in exact arithmetic; to
    # floating-point roundoff — ~1e-7 relative in fp32).  It is NOT column-
    # water-vapour (CWV = sum_c A_c dp_c q_c/g) conserving: across a surface-
    # pressure gradient an unweighted del2 redistributes a little water mass
    # (bias set by the humidity–terrain correlation), so this is an explicitly
    # NON-conservative grid-scale filter — monitor the water budget if used in
    # long runs.  Unweighted (not dp-weighted) is REQUIRED: the default hybrid
    # coordinate's surface-layer dp goes <= 0 for p_s below ~2/3 p_ref over
    # high terrain, which a dp-weighted form would divide by (Inf/NaN).  It is
    # a del2 (∇²), not the FV lanes' scale-selective del4 (∇⁴): del2 is
    # monotone under the explicit guard (hence the exact per-level integral)
    # but damps resolved gradients more broadly — use a gentle coefficient.
    # MPAS-only: refused on other discretizations (validate_strict).
    mpas_qv_smooth_del2_m2s: float = 0.0   # del2 diffusivity [m^2/s]; ~1e5-1e6 typical at 240 km
    # Scale-SELECTIVE companion to the del2 above, added 2026-09-11 after the
    # del2's cost was measured: across thirteen arms differing only in
    # mpas_qv_smooth_del2_m2s, the humidity field's structure-function growth
    # exponent falls monotonically from 1.56 at 2e5 to 0.46 at 1.1e4 — i.e. the
    # del2 IS holding grid-scale noise down, but it has to be strong enough to
    # flatten the resolved gradients that set tropical cloud cover along with
    # it.  A biharmonic's eigenvalue is exactly minus the SQUARE of the del2's
    # on every mesh mode, so its damping ratio between any two scales is the
    # del2's squared: measured on the subdivision-6 mesh, 2.47x between 240 and
    # 479 km becomes 6.08x.  (The continuum figures 4 and 16 do not apply to
    # the discrete operator.)  It is NOT monotone, so the q>=0 floor is
    # load-bearing when it is on and the per-level integral is conserved only
    # up to what that floor clips; the driver enforces nu4*dt*g_max^2 <= 0.5 at
    # setup.  MPAS-only, same as the del2.  Default 0.0 pending the controlled
    # pair that picks the coefficient — see the arm before changing it.
    mpas_qv_smooth_del4_m4s: float = 0.0   # del4 diffusivity [m^4/s]
    # Prognostic sea-ice skin temperature on the MPAS lane (Semtner 1976
    # zero-layer conduction + slab thermal inertia; forcing/surface_utils
    # helper).  The prescribed-SST anchor otherwise pins ice-covered cells
    # at the CONSTANT T_ice (271.35 K) year-round — the +8.6 K polar tas
    # warm bias of the first ClimateEval scorecard (real central-Arctic
    # winter skin ~245-250 K).  Default OFF = byte-identical.  Needs
    # radiation != "none" (the skin integrates the exported surface
    # fluxes); refused on non-MPAS lanes (their land/ice tiles own the
    # surface temperature).
    mpas_ice_skin_prognostic: bool = False
    mpas_ice_thickness_m: float = 2.0      # climatological ice slab thickness [m]
    # Hines (1997) non-orographic GWD launch amplitude + saturation flux cap.
    # Reached through gwd_config_for on EVERY lane (like the mcfarlane_*
    # scalars above, which were silently inert on every production path until
    # that resolver existed).  The low-level extratropical westerlies are the
    # observable lever: hines deposits momentum that decelerates them.
    # Appended at the tuple END to preserve the positional ABI.
    hines_total_rms_wind: float = 2.0           # HinesConfig.total_rms_wind [m/s]
    hines_Fmax: float = 0.1                     # HinesConfig.Fmax [Pa]
    # HinesConfig.launch_p [Pa]; 0.0 = unset = legacy SURFACE launch.
    # A non-orographic wave launched at the surface is born supersaturated
    # in the weakly stratified BL and breaks at its own launch level
    # (measured: 55% of its momentum deposited below 1 km).
    hines_launch_p: float = 0.0
    e3sm_cam_source: str = "orographic"         # E3SMCAMConfig.source
    e3sm_cam_pgwv: int = 0                      # phase-speed half-width (waves either side of c0)
    e3sm_cam_effgw: float = 0.125               # E3SMCAMConfig.effgw [dimensionless]
    e3sm_cam_taubgnd: float = 1.5e-3            # E3SMFrontalConfig.taubgnd [Pa]
    e3sm_cam_c0: float = 30.0                   # E3SMFrontalConfig.c0 [m/s]
    e3sm_cam_launch_p: float = 5.0e4            # E3SMFrontalConfig.launch_p [Pa]
    e3sm_cam_latitude_taper: bool = False       # E3SMFrontalConfig.latitude_taper:
                                                # OFF, E3SM's unstructured branch
                                                # (our grids); see that field's docs
    # CAM6-suite additions (2026-09-21).  ``e3sm_cam_source`` also accepts a
    # ``+``-joined set ("orographic+frontal+convective").  None = the kernel's
    # single ``e3sm_cam_effgw`` (byte-identical); CAM6 f09 sets effgw_cm=1.0,
    # effgw_beres_dp=0.4.  ``e3sm_cam_beres_variant``: "e3sm" (default) or
    # "cam6" (gw_convect.F90 end-off shift, real storm speed, interface
    # source level; see gwd_config_for).  ``e3sm_cam_mfcc_table_path``: the
    # offline Beres table (newmfspectra40_dc25.nc); "" = stand-in spectrum.
    e3sm_cam_effgw_cm: float | None = None      # E3SMFrontalConfig.effgw
    e3sm_cam_effgw_beres: float | None = None   # E3SMBeresConfig.effgw
    e3sm_cam_frontgfc: float | None = None      # E3SMFrontalConfig.frontgfc
                                                # [K^2/(m^2 s)]; None = kernel
                                                # default 1.25e-15; CAM6 f09 3.0e-15
    e3sm_cam_beres_variant: str = "e3sm"        # "e3sm" | "cam6" (also sets
                                                # CAM's min_hdepth 1 km + row lookup)
    e3sm_cam_mfcc_table_path: str = ""          # E3SMBeresConfig.mfcc_table_path
    e3sm_cam_dttke_intrinsic: bool = False      # E3SMCAMConfig.dttke_use_intrinsic
                                                # (CAM6 heating form; E3SM-3.0.1 off)
    # Appended at the tuple END to preserve the positional ABI (codex
    # 2026-07-27 flavor review, Major 1).
    morrison_flavor: str = "mg"                 # MorrisonConfig.morrison_flavor:
                                                # "mg" (E3SM MG, GCM default) |
                                                # "sam" (gSAM M2005 anvil tune)
    # Flux law the SLAB-land skin energy balance debits at the land-air
    # interface (physics_pipeline._step_slab_land):
    #   "legacy_dual" (default, byte-identical): the slab debits its OWN
    #       constant-C_H/C_E no-stability bulk fluxes while the atmosphere's
    #       turbulence scheme debits stability-dependent surface-layer fluxes
    #       (compute_surface_fluxes, config.surface) from the SAME interface —
    #       two different flux laws, measured same-state mismatch
    #       +75..+152 W/m^2 (a spurious skin heat source; energy is NOT
    #       conserved at the interface).  Kept as the default only for
    #       reproducibility of existing runs.
    #   "unified": the slab consumes the SAME sensible+latent flux law the
    #       atmosphere side applies (the turbulence scheme's surface layer on
    #       the blended surface; the tiled land-MOST law when surface_tiled),
    #       so both sides are ONE function of ONE state — the flux-LAW split
    #       is gone.  What remains is the semi-implicit time-discretization
    #       term max(d(SH+LE)/dT,0)*dT_skin (vanishes as the skin nears
    #       equilibrium; NOT a systematic two-law bias, but the same ORDER as
    #       the removed defect at a 4-hourly radiation cadence — see
    #       _step_slab_land's measured table; a shorter rad_update_steps or a
    #       larger C_land shrinks it).  ENERGY-ONLY scope: the soil-water
    #       bucket still drains its legacy beta*C_E bulk evaporation (the
    #       pre-existing non-tiled water-budget non-closure, warned at driver
    #       setup, is unchanged).  Requires an active turbulence scheme and
    #       an active slab land tile on the pipeline lanes (validate_strict).
    # Appended at the tuple END to preserve the positional ABI.
    land_interface_flux: str = "legacy_dual"

    # Free-atmosphere diffusivity floor override [m^2/s] for schemes carrying a
    # ``kvf_min`` field (holtslag_boville).  None = scheme default (byte-
    # identical).  CAUSALITY-PROBE knob for the polar-night stable-transport
    # runaway (interior K floors at kvf_min under a surface inversion while the
    # LW deficit is tens of W/m^2); the paper-grade remedy is an interior
    # stable-tail selector, not this floor.  Appended at the tuple END to
    # preserve the positional ABI.
    hb_kvf_min: float | None = None
    # IFS deep entrainment/detrainment base rates (plume-mixing control):
    # epsilon_deep up = more dilution, weaker and shallower plumes;
    # delta_deep down = less condensate leaked to the anvil, more left in
    # the plume to rain out.  Wired 2026-08-13; before this the deep
    # entrainment rate could not be set from any MIP driver at all.
    # APPENDED, not inserted beside the other bechtold_* fields: this is a
    # NamedTuple, so a mid-struct insertion silently reassigns every later
    # positional argument.
    bechtold_epsilon_deep: float = 1.75e-3  # BechtoldConfig.epsilon_deep [1/m]
    bechtold_delta_deep: float = 0.75e-4    # BechtoldConfig.delta_deep [1/m]
    # Scale on the LAND branch of the diurnal-cycle CAPE subtraction. The IFS
    # value assumes a ~10 km mesh; 1.0 reproduces it, 0.0 removes the land
    # branch while leaving the ocean branch alone -- which the on/off flag
    # cannot do, because it disables both.
    bechtold_capdcycl_land_tau_scale: float = 1.0
    bechtold_subcloud_evap_scale: float = 1.0
    bechtold_rhebc_land: float = 0.75
    bechtold_rhebc_land_deep: float = 0.70

    # --- CAM6 macro/micro sub-cycle + MG2 CFL sedimentation --------------
    # APPENDED AT THE TUPLE END to preserve the positional ABI (a field
    # inserted mid-tuple silently re-binds every positional construction and
    # every pickle written before it).
    # CAM6 ``cld_macmic_num_steps``: turbulence (macrophysics) and
    # microphysics run N times sequentially at dt_phys/N inside each physics
    # step, after the convective increment (CAM tphysbc macmic loop).  1 =
    # the parallel split (byte-identical).  MPAS lane.
    cld_macmic_num_steps: int = 1
    # MorrisonConfig.sed_cfl_substeps: MG2 CFL sub-stepped sedimentation
    # (default ON, user 2026-09-22; False = the legacy one-pass form).
    morrison_sed_cfl_substeps: bool = True
    # MorrisonConfig.sed_cfl_substeps_max: static bound of the sub-step loop.
    # The loop cost is LINEAR in it (2048 columns, CPU x64, whole Morrison
    # call: one pass 10 ms, 16 -> 24 ms, 96 -> 59 ms, 256 -> 131 ms), and the
    # required counts are 8 on production sigma-36 at 112.5 s and 90 on CAM
    # L32 at 600 s, so a deck should size it instead of paying for 256.
    morrison_sed_cfl_substeps_max: int = 256
    # MorrisonConfig.sed_cfl_substeps_strict: runtime error when a column
    # needs more sub-steps than the cap.
    morrison_sed_cfl_substeps_strict: bool = False

    def _liquid_partition_resolved(self) -> bool:
        """Is CLUBB's cloud-liquid exchange selected, by ANY route?

        Not the experiment flag alone: an authoritative ``turbulence_override``
        can carry ``CLUBBConfig(liquid_partition=True)`` without it ever being
        set, and that route reached a validated, built model with both
        radiative condensate floors still active (codex).

        OPEN BEFORE THIS IS ENABLED ANYWHERE, both from the round-3 reviews and
        neither closed here:

        * the optics cover accounting.  The arm this lever targets carries MORE
          cloud cover than the baseline (61.5 % against 52.2 % over the arms'
          identical first 20 days) and still reflects 14.5 W/m2 LESS sunlight,
          with clear-sky fluxes matching to 0.3.  Under the standard accounting
          more cover at fixed grid-mean water should reflect MORE, so either the
          liquid mass really is the deficit (which this lever addresses) or the
          radiation is fed an in-cloud optical depth and treats it as a
          grid-mean one (which it does not).  Read where cover enters optical
          depth versus albedo before crediting any radiative effect to this.
        * ``clubb_dt`` reconciliation across the host sub-steps, which becomes
          load-bearing because the guard above forces the macro/micro sub-cycle
          on wherever this is enabled.

        A future change that turns this on has to confront both.
        """
        if self.clubb_liquid_partition:
            return True
        _ov = self.turbulence_override
        if _ov is not None and getattr(_ov, "scheme", None) == "clubb":
            return bool(getattr(getattr(_ov, "clubb", None),
                                "liquid_partition", False))
        return False

    def validate_strict(self) -> None:
        """Raise ValueError for invalid parameter values.

        Called before simulation start to catch configuration errors
        early, before JIT compilation or data loading.
        """
        g = self.grid
        d = self.dycore
        errors: list[str] = []
        # Owner decision 2026-08-16: conserving form always, and any
        # NON-conserving form must ANNOUNCE itself when invoked. These are
        # warnings, not errors — the legacy/exception paths stay selectable
        # for bit-comparison and for the documented MSE-vs-water trade-off.
        import logging as _logging
        _wlog = _logging.getLogger(__name__)
        if not d.mpas_conservative_tracer_clamp:
            _wlog.warning(
                "NON-CONSERVING form selected: "
                "mpas_conservative_tracer_clamp=False restores the plain "
                "max(q, 0) tracer clamp, which CREATES mass at every "
                "transport undershoot (+30 kg/m2/yr of water measured on "
                "the AMIP century). Legacy bit-comparison mode only.")
        if self.energy_consistent_moisture_clip:
            _wlog.warning(
                "NON-CONSERVING form selected: "
                "energy_consistent_moisture_clip=True conserves "
                "moist static energy but CREATES the clipped vapour "
                "(water is NOT conserved by this floor) — the documented "
                "exception to 'conserving form always'.")
        if g.resolution <= 0:
            errors.append(f"grid.resolution must be > 0, got {g.resolution}")
        if g.nlev <= 0:
            errors.append(f"grid.nlev must be > 0, got {g.nlev}")
        if g.p_top_Pa <= 0:
            errors.append(f"grid.p_top_Pa must be > 0, got {g.p_top_Pa}")
        if g.vertical_coord == "sigma":
            if not (0.0 < g.sigma_top < 1.0):
                errors.append(
                    f"grid.sigma_top must be in (0, 1) on the sigma lane (got {g.sigma_top})")
            if not (g.sigma_top < g.sigma_refine < 1.0):
                errors.append(
                    f"grid.sigma_refine must lie in (sigma_top, 1) (got {g.sigma_refine})")
            if not (g.sigma_refine_width > 0.0 and math.isfinite(g.sigma_refine_width)):
                errors.append(
                    f"grid.sigma_refine_width must be > 0 (got {g.sigma_refine_width})")
            if g.sigma_layout not in ("standard", "l30_trop_logstrat"):
                errors.append(
                    "grid.sigma_layout must be 'standard' or 'l30_trop_logstrat' "
                    f"(got {g.sigma_layout!r})")
            elif g.sigma_layout == "l30_trop_logstrat":
                if g.nlev < 28:
                    errors.append(
                        f"grid.sigma_layout 'l30_trop_logstrat' needs nlev >= 28 (got {g.nlev})")
                if not (0.0 < g.sigma_top <= 0.05):
                    errors.append(
                        "grid.sigma_layout 'l30_trop_logstrat' needs 0 < sigma_top <= 0.05 "
                        f"(grids.vertical.L30_LOGSTRAT_SIGMA_TOP_MAX; got {g.sigma_top})")
                if g.tropopause_refine != 1.0:
                    errors.append(
                        "grid.sigma_layout 'l30_trop_logstrat' keeps the L30 troposphere and "
                        f"cannot combine with tropopause_refine={g.tropopause_refine}")
            if g.p_top_Pa != 200.0 or g.stretching != 2.0:
                errors.append(
                    "grid.p_top_Pa and grid.stretching are hybrid-only fields and are inert "
                    "on the sigma lane; leave them at their defaults (200.0, 2.0) and set "
                    "grid.sigma_top instead")
            if g.transition_exponent != 3:
                errors.append(
                    "grid.transition_exponent is a hybrid-only field and is inert on the "
                    f"sigma lane; leave it at its default 3 (got {g.transition_exponent})")
        elif (g.vertical_coord == "hybrid"
                and g.transition_exponent not in (2, 3)):
            # 1 is pure sigma spelled as a hybrid (use vertical_coord='sigma'
            # instead, which is the tested lane); >3 makes the negative-mass
            # threshold worse, not better.
            errors.append(
                "grid.transition_exponent must be 2 or 3 on the hybrid lane "
                f"(got {g.transition_exponent}); 2 admits p_s down to ~498 hPa "
                "(~5870 m of orography), 3 only to ~664 hPa (~3450 m). For a "
                "pure sigma coordinate use vertical_coord='sigma'.")
        elif g.vertical_coord == "hybrid" and g.sigma_top != 0.01:
            errors.append(
                "grid.sigma_top is inert on the hybrid lane, which uses p_top_Pa/"
                f"stretching; leave it at its default 0.01 (got {g.sigma_top})")
        elif g.vertical_coord == "hybrid" and g.sigma_layout != "standard":
            errors.append(
                f"grid.sigma_layout={g.sigma_layout!r} is inert on the hybrid lane")
        elif g.vertical_coord == "cam_l32":
            # The CAM6 table fixes every interface: nlev, top and spacing are
            # not free, and the analytic-hybrid / sigma knobs would be inert.
            if g.nlev != 32:
                errors.append(
                    f"grid.vertical_coord='cam_l32' is the CAM6 32-level table; nlev must be 32 (got {g.nlev})")
            if (g.p_top_Pa != 200.0 or g.stretching != 2.0 or g.sigma_top != 0.01
                    or g.sigma_layout != "standard" or g.tropopause_refine != 1.0):
                errors.append(
                    "grid.p_top_Pa/stretching/sigma_top/sigma_layout/tropopause_refine are inert "
                    "on vertical_coord='cam_l32' (the table fixes them); leave them at their defaults")
        elif g.vertical_coord not in ("sigma", "hybrid"):
            errors.append(
                f"grid.vertical_coord must be one of ('sigma', 'hybrid', 'cam_l32'); got {g.vertical_coord!r}")
        # Tropopause refinement: 1.0 = uniform.  The upper bound is NOT a
        # vertical-CFL limit — the first-order-upwind vertical advective CFL
        # is only 0.26 at refine=3 / dt=75 s / omega=5 Pa/s and 0.39 at
        # refine=6, the index-space nu_vert4_T filter does not scale with
        # dsigma at all, and both GWD schemes cap acceleration
        # thickness-independently (measured 2026-07-25).  The binding
        # criterion is GRID SMOOTHNESS: the max adjacent-layer thickness ratio
        # is 1.00 at refine=1, 1.62 at refine=3 and 2.41 at refine=6, whereas
        # every grid this model has run successfully is <= 1.07 and
        # operational practice keeps it <~ 1.2.  A sharp stretch transition
        # reflects resolved vertical waves and worsens the (pre-existing)
        # mismatch between the Simmons-Burridge pressure at which Phi_k is
        # defined and the arithmetic midpoint at which T/q/physics live.
        # 3.0 is the largest value with an actual stability arm behind it;
        # raising this bound requires a new one, not a wider constant.
        if not (1.0 <= g.tropopause_refine <= 3.0
                and math.isfinite(g.tropopause_refine)):
            errors.append(
                f"grid.tropopause_refine must be finite in [1, 3] "
                f"(1 = uniform); got {g.tropopause_refine}")
        if g.tropopause_refine != 1.0 and g.vertical_coord != "sigma":
            errors.append(
                f"grid.tropopause_refine={g.tropopause_refine} only applies "
                f"to the sigma coordinate; vertical_coord="
                f"{g.vertical_coord!r} has its own `stretching` and would "
                "silently ignore it.")
        if d.dt <= 0:
            errors.append(f"dycore.dt must be > 0, got {d.dt}")
        if d.fv3_duo_windows is not None:
            if d.discretization != "fv3_duo":
                errors.append(
                    "dycore.fv3_duo_windows is only meaningful with "
                    f"dycore.discretization='fv3_duo', got {d.discretization!r}")
            if not isinstance(d.fv3_duo_windows, int) or d.fv3_duo_windows < 2:
                errors.append(
                    "dycore.fv3_duo_windows must be an int >= 2 (kt, for "
                    f"6*kt*kt ranks), got {d.fv3_duo_windows!r}")
            if d.fv3_duo_window_pad is None:
                errors.append(
                    "dycore.fv3_duo_window_pad is REQUIRED with "
                    "dycore.fv3_duo_windows (the pad is a measured per-deck "
                    "halo width, never defaulted)")
            elif not isinstance(d.fv3_duo_window_pad, int) or d.fv3_duo_window_pad < 1:
                errors.append(
                    "dycore.fv3_duo_window_pad must be an int >= 1, got "
                    f"{d.fv3_duo_window_pad!r}")
        elif d.fv3_duo_window_pad is not None:
            errors.append(
                "dycore.fv3_duo_window_pad given without dycore.fv3_duo_windows")
        if d.fv3_duo_column_lane:
            if d.discretization != "fv3_duo":
                errors.append(
                    "dycore.fv3_duo_column_lane needs "
                    f"dycore.discretization='fv3_duo', got {d.discretization!r}")
            if d.fv3_duo_windows is not None:
                errors.append(
                    "dycore.fv3_duo_column_lane runs on six faces; the window "
                    "layout is certification rung 7 (drop fv3_duo_windows)")
        if d.fv3_duo_fill and d.discretization != "fv3_duo":
            errors.append(
                "dycore.fv3_duo_fill is the fv3_duo remap's fillz; got "
                f"dycore.discretization={d.discretization!r}")
        if d.hyperdiff_scale < 0:
            errors.append(f"dycore.hyperdiff_scale must be >= 0, got {d.hyperdiff_scale}")
        if d.div_damp_scale < 0:
            errors.append(f"dycore.div_damp_scale must be >= 0, got {d.div_damp_scale}")
        # NaN/inf must die here: the scale multiplies a coefficient that goes
        # straight into a momentum tendency, so a non-finite value does not
        # raise anywhere downstream -- it silently turns the whole wind field
        # non-finite on the first step (codex review 2026-09-22).
        _dd4 = getattr(d, "mpas_div_damp4_scale", 0.0)
        if (not isinstance(_dd4, (int, float)) or isinstance(_dd4, bool)
                or not math.isfinite(_dd4) or _dd4 < 0):
            errors.append(
                "dycore.mpas_div_damp4_scale must be a finite number >= 0, "
                f"got {_dd4!r}")
        # Dynamical-core axis membership.  Mirror the gate in
        # ``atmosphere.dynamics.resolve_solver_name`` so a typo fails here, at
        # config-validation time, instead of deep in the solver factory at JIT.
        # Deferred import (driver -> atmosphere is the allowed direction; by the
        # time validate_strict runs the dynamics package is loaded anyway).
        from legoesm.atmosphere.dynamics import (
            DISCRETIZATION_OPTIONS,
            DYNAMICS_OPTIONS,
        )
        if d.model_type not in DYNAMICS_OPTIONS:
            errors.append(
                f"dycore.model_type must be one of {DYNAMICS_OPTIONS}, "
                f"got {d.model_type!r}"
            )
        elif d.model_type == "nonhydrostatic" and (g.sigma_top != 0.01
                                                   or g.sigma_layout != "standard"):
            errors.append(
                "grid.sigma_top and grid.sigma_layout are inert on the "
                "nonhydrostatic lane, which uses a height coordinate; leave "
                f"them at their defaults 0.01/'standard' (got sigma_top="
                f"{g.sigma_top}, sigma_layout={g.sigma_layout!r})")
        if d.discretization not in DISCRETIZATION_OPTIONS:
            errors.append(
                f"dycore.discretization must be one of {DISCRETIZATION_OPTIONS}, "
                f"got {d.discretization!r}"
            )
        # NOTE (2026-07-23 fix-forward): the dycore.pgf_scheme membership
        # check (3fa76cc7b) referenced a field added by a3b450b7f, which is
        # on ap/amip-cmip6-integration and NOT yet on main — every
        # validate_strict() call died on AttributeError.  Re-add the check
        # (and its bogus-teeth coverage in test_validate_strict_coverage)
        # together with the a3b450b7f DycoreConfig.pgf_scheme field.
        _valid_precisions = ("fp32", "fp64", "mixed", "mixed_fp64_storage")
        if self.precision not in _valid_precisions:
            errors.append(
                f"precision must be one of {_valid_precisions}, got {self.precision!r}"
            )
        _valid_distributed_modes = ("mpi", "spmd")
        if self.distributed_mode not in _valid_distributed_modes:
            errors.append(
                f"distributed_mode must be one of {_valid_distributed_modes}, "
                f"got {self.distributed_mode!r}"
            )
        if (self.distributed and self.distributed_mode == "spmd"
                and g.grid_type != "cubed_sphere"):
            errors.append(
                "distributed_mode='spmd' supports only "
                "grid.grid_type='cubed_sphere' (got "
                f"{g.grid_type!r}): the lat-lon/MPAS distributed paths "
                "arm the mpi4jax halo backend, which must never coexist "
                "with jax.distributed collectives in one program"
            )
        if self.distributed and self.distributed_mode == "spmd":
            # Checkpointing IS supported (cs_spmd step 5a):
            # ``save_checkpoint`` gathers the face-sharded state to a host
            # replica on EVERY process (collective process_allgather —
            # see ``_gather_spmd_tree_to_host``) and process 0 writes the
            # standard single-file restart.
            # Diagnostics are FULLY supported (steps 5b+5c): perf mode
            # runs ``collect_lightweight`` (SPMD-global jnp reductions,
            # replicated scalar results); the full ``collect()``
            # (snapshots/profiles/monthly/CMIP — cmip_output or
            # diagnostics_perf_mode='never') gathers the sharded fields
            # to host replicas on every process first.  All flush/save
            # sites are root-gated via ``_mpi_rank = jax.process_index()``.
            pass
        if self.enable_tiled_dycore:
            # P4 sub-face-tiled cube dynamics (single-controller multi-device;
            # (6,kt,kt) mesh).  Cube-only; mpi4jax-distributed runs have no
            # tiled device mesh (the driver helper also fails loudly there).
            if g.grid_type != "cubed_sphere":
                errors.append(
                    "enable_tiled_dycore=True requires "
                    f"grid.grid_type='cubed_sphere' (got {g.grid_type!r}): "
                    "the tiled step is the cube sub-face decomposition"
                )
            if self.distributed and self.distributed_mode == "mpi":
                errors.append(
                    "enable_tiled_dycore=True is a device-mesh (SPMD) path "
                    "and cannot run under the mpi4jax replicated-faces mode "
                    "(distributed_mode='mpi'); use single-process multi-GPU "
                    "or distributed_mode='spmd'"
                )

        if self.enable_latlon_spmd:
            # Single-process multi-device lat-band path (NOT distributed_mode).
            if g.grid_type != "latlon":
                errors.append(
                    "enable_latlon_spmd=True requires grid.grid_type='latlon' "
                    f"(got {g.grid_type!r}): the lat-band decomposition is the "
                    "lat-lon C-grid twin of the ocean SPMD step"
                )
            if self.distributed:
                errors.append(
                    "enable_latlon_spmd=True is the SINGLE-PROCESS multi-device "
                    "path and is mutually exclusive with distributed=True "
                    "(MPI / multi-controller); use one or the other"
                )
            # The n_lat % n_devices divisibility constraint is checked at
            # runtime in ModelDriver._latlon_spmd_mesh against the BUILT
            # LatLonGrid (GridConfig carries only ``resolution``, not the
            # derived n_lat/n_lon), so a wrong device count fails LOUDLY there.
        if self.latlon_spmd_compiled_segments and not self.enable_latlon_spmd:
            errors.append(
                "latlon_spmd_compiled_segments=True requires "
                "enable_latlon_spmd=True: the compiled-scan segment lane is a "
                "mode OF the lat-band SPMD run loop (run_atm_latlon_spmd) and "
                "is a silent no-op on every other path"
            )
        if self.days <= 0:
            errors.append(f"days must be > 0, got {self.days}")
        if self.seed < 0:
            errors.append(f"seed must be >= 0, got {self.seed}")
        if (self.land_update_seconds < 0
                or not math.isfinite(self.land_update_seconds)):
            errors.append(
                f"land_update_seconds must be a finite value >= 0 "
                f"(0 = every step), got {self.land_update_seconds}")
        if self.land_update_seconds > 0 and not self.use_multilayer_land:
            errors.append(
                "land_update_seconds > 0 requires use_multilayer_land: the "
                "slab land has no held-flux cadence — the knob would be "
                "silently inert.")
        if (self.land_update_seconds > 0
                and self.grid.grid_type not in ("mpas", "voronoi")):
            errors.append(
                "land_update_seconds > 0 is implemented only on the MPAS "
                f"lane; grid_type={self.grid.grid_type!r} would silently "
                "ignore it.")
        if self.forcing_update_days <= 0:
            errors.append(
                f"forcing_update_days must be > 0, got "
                f"{self.forcing_update_days}"
            )
        if self.sbm_cape_threshold < 0:
            errors.append(
                f"sbm_cape_threshold must be >= 0, got {self.sbm_cape_threshold}"
            )
        if self.bechtold_cape_threshold < 0:
            errors.append(
                f"bechtold_cape_threshold must be >= 0, got "
                f"{self.bechtold_cape_threshold}"
            )
        if self.bechtold_subsidence_solve not in ("implicit_flux", "advective"):
            errors.append(
                f"bechtold_subsidence_solve must be one of "
                f"('implicit_flux', 'advective'), got "
                f"{self.bechtold_subsidence_solve!r}"
            )
        if self.bechtold_rain_vapor_sink not in ("formation", "vapour_mass"):
            errors.append(
                f"bechtold_rain_vapor_sink must be one of "
                f"('formation', 'vapour_mass'), got "
                f"{self.bechtold_rain_vapor_sink!r}"
            )
        if (self.convective_precip_efficiency is not None
                and not (0.0 <= self.convective_precip_efficiency <= 1.0)):
            errors.append(
                f"convective_precip_efficiency must be in [0, 1] or None, got "
                f"{self.convective_precip_efficiency}"
            )
        if self.convective_precip_split not in ("constant", "autoconversion"):
            errors.append(
                "convective_precip_split must be 'constant' or 'autoconversion', "
                f"got {self.convective_precip_split!r}"
            )
        if not (0.0 < self.autoconv_q_c_crit <= 1.0e-2):
            errors.append(
                f"autoconv_q_c_crit must be in (0, 1e-2] kg/kg, got "
                f"{self.autoconv_q_c_crit}"
            )
        if not (0.0 <= self.autoconv_pe_max <= 1.0):
            errors.append(
                f"autoconv_pe_max must be in [0, 1], got {self.autoconv_pe_max}"
            )
        if self.bechtold_conv_top_pa <= 0.0:
            errors.append(
                f"bechtold_conv_top_pa must be > 0 Pa (the convective-top gate "
                f"cutoff), got {self.bechtold_conv_top_pa}"
            )
        for _f, _lo, _hi in (
            # Ceiling raised with the scheme spec (see BechtoldConfig
            # __param_spec__): the production value sat on the old bound.
            ("bechtold_rprcon", 3.5e-4, 1.4e-2),
            ("bechtold_M_b_max", 0.02, 0.15),
            ("bechtold_epsilon_deep", 7.0e-4, 4.2e-3),
            ("bechtold_delta_deep", 3.0e-5, 1.8e-4),
            ("bechtold_dnoprc", 7.5e-5, 1.2e-3),
            ("bechtold_epsilon_deep", 5.775e-04, 3.5e-03),
            ("bechtold_delta_deep", 2.475e-05, 2.25e-04),
            ("bechtold_capdcycl_land_tau_scale", 0.0, 2.0),
            ("bechtold_subcloud_evap_scale", 0.1, 4.0),
            ("bechtold_rhebc_land", 0.5, 1.0),
            ("bechtold_rhebc_land_deep", 0.5, 1.0),
            ("bechtold_downdraft_evap", 0.0, 0.5),
            ("bechtold_downdraft_alpha", 0.0, 0.9),
            ("bechtold_downdraft_rh_min", 0.0, 1.0),
            ("bechtold_downdraft_entrain_rate", 1.0e-4, 2.0e-3),
            ("bechtold_downdraft_detrain_scale_m", 100.0, 3000.0),
        ):
            _v = getattr(self, _f)
            if not (_lo <= _v <= _hi):
                errors.append(f"{_f}={_v!r} out of range [{_lo}, {_hi}]")
        if self.physics_parameterization not in ("none", "ml"):
            errors.append(
                "physics_parameterization must be 'none' or 'ml', "
                f"got {self.physics_parameterization!r}"
            )
        if self.physics_parameterization_hidden_dim <= 0:
            errors.append(
                "physics_parameterization_hidden_dim must be > 0, "
                f"got {self.physics_parameterization_hidden_dim}"
            )
        if self.physics_parameterization_layers <= 0:
            errors.append(
                "physics_parameterization_layers must be > 0, "
                f"got {self.physics_parameterization_layers}"
            )
        # Radiation membership (reconciled: physics_pipeline now raises on
        # unknown and builds an explicit zero-radiation fn for "none", matching
        # this accepted set = _RADIATION_BUILDERS keys).
        _valid_radiation = VALID_RADIATION
        if self.radiation not in _valid_radiation:
            errors.append(
                f"radiation must be one of {_valid_radiation}, "
                f"got {self.radiation!r}"
            )
        _valid_cloud_schemes = VALID_CLOUD_SCHEMES
        if self.cloud_scheme not in _valid_cloud_schemes:
            errors.append(
                f"cloud_scheme must be one of {_valid_cloud_schemes}, "
                f"got {self.cloud_scheme!r}"
            )
        _valid_diag_condensate = ("constant", "adiabatic")
        if self.cloud_diagnostic_condensate_scheme not in _valid_diag_condensate:
            errors.append(
                f"cloud_diagnostic_condensate_scheme must be one of "
                f"{_valid_diag_condensate}, "
                f"got {self.cloud_diagnostic_condensate_scheme!r}"
            )
        _valid_inhom = ("constant", "two_region")
        if self.cloud_optics_inhomogeneity not in _valid_inhom:
            errors.append(
                f"cloud_optics_inhomogeneity must be one of {_valid_inhom}, "
                f"got {self.cloud_optics_inhomogeneity!r}"
            )
        _valid_cover = ("none", "two_column")
        if self.cloud_partial_coverage_optics not in _valid_cover:
            errors.append(
                f"cloud_partial_coverage_optics must be one of {_valid_cover}, "
                f"got {self.cloud_partial_coverage_optics!r}"
            )
        _valid_overlap = ("none", "max_random")
        if self.cloud_vertical_overlap_optics not in _valid_overlap:
            errors.append(
                f"cloud_vertical_overlap_optics must be one of "
                f"{_valid_overlap}, got {self.cloud_vertical_overlap_optics!r}"
            )
        _valid_sat = ("liquid", "mixed_phase")
        if self.cloud_saturation_scheme not in _valid_sat:
            errors.append(
                f"cloud_saturation_scheme must be one of {_valid_sat}, "
                f"got {self.cloud_saturation_scheme!r}"
            )
        if (self.cloud_partial_coverage_optics != "none"
                and self.cloud_vertical_overlap_optics != "none"):
            # Caught here as well as at the scheme, so a bad config fails at
            # startup rather than inside the first radiation call.
            errors.append(
                "cloud_partial_coverage_optics and "
                "cloud_vertical_overlap_optics are mutually exclusive (both "
                "correct partial cloud coverage); got "
                f"{self.cloud_partial_coverage_optics!r} and "
                f"{self.cloud_vertical_overlap_optics!r}"
            )
        if not 1 <= int(self.cloud_n_subcolumns) <= 64:
            errors.append(
                f"cloud_n_subcolumns must be in [1, 64], got "
                f"{self.cloud_n_subcolumns}"
            )
        # External-forcing source selectors.  The driver activates each channel
        # with a ``cfg.<field> == "external"``-style equality gate
        # (model_driver._setup_external_forcing), so a typo'd value is NOT an
        # unknown-scheme error at the loader — it silently deactivates the
        # channel (an AMIP run with ``ozone_forcing="externl"`` runs the static
        # fallback profile).  Membership-check them here so the failure is
        # fail-early and loud (dispatch-hardening).
        _valid_ozone_forcing = ("inline", "external", "off")
        if self.ozone_forcing not in _valid_ozone_forcing:
            errors.append(
                f"ozone_forcing must be one of {_valid_ozone_forcing}, "
                f"got {self.ozone_forcing!r}"
            )
        # "mls" (SAM RCEMIP MLS climatology) is implemented in
        # _compute_ozone_vmr and reachable from YAML/programmatic configs;
        # "ml" is NOT accepted here — it needs T/lon/ridge-weight plumbing
        # the ExperimentConfig path does not carry.
        _valid_ozone_source = ("standard", "analytical", "mls", "none")
        if self.ozone_source not in _valid_ozone_source:
            errors.append(
                f"ozone_source must be one of {_valid_ozone_source}, "
                f"got {self.ozone_source!r}"
            )
        _valid_ghg_forcing = ("constant", "external")
        if self.ghg_forcing not in _valid_ghg_forcing:
            errors.append(
                f"ghg_forcing must be one of {_valid_ghg_forcing}, "
                f"got {self.ghg_forcing!r}"
            )
        _valid_aerosol_forcing = ("off", "external")
        if self.aerosol_forcing not in _valid_aerosol_forcing:
            errors.append(
                f"aerosol_forcing must be one of {_valid_aerosol_forcing}, "
                f"got {self.aerosol_forcing!r}"
            )
        _valid_solar_source = ("constant", "file", "spectral_file")
        if self.solar_source not in _valid_solar_source:
            errors.append(
                f"solar_source must be one of {_valid_solar_source}, "
                f"got {self.solar_source!r}"
            )
        _valid_band_order = ("auto", "as_is", "rrtmg_sw")
        if self.solar_spectral_band_order not in _valid_band_order:
            errors.append(
                f"solar_spectral_band_order must be one of {_valid_band_order}, "
                f"got {self.solar_spectral_band_order!r}"
            )
        _valid_datasets = ("cobe", "hadisst", "custom", "analytical")
        if self.dataset not in _valid_datasets:
            errors.append(
                f"dataset must be one of {_valid_datasets}, "
                f"got {self.dataset!r}"
            )
        # CMIP experiment id: consumed by ghg_at_year at driver setup; reject
        # unknown ids (and the fixed-forcing x external-file contradiction)
        # before a run spends JIT time.  ``""`` = no experiment override.
        if self.experiment:
            from legoesm.forcing.experiments import EXPERIMENT_TEMPLATES
            if self.experiment not in EXPERIMENT_TEMPLATES:
                errors.append(
                    f"experiment must be one of "
                    f"{tuple(sorted(EXPERIMENT_TEMPLATES))} (or ''), "
                    f"got {self.experiment!r}"
                )
            elif (EXPERIMENT_TEMPLATES[self.experiment].forcing_type == "fixed"
                    and self.ghg_forcing == "external"):
                # The external annual GHG file takes precedence over the
                # experiment scalars in _precompute_external_forcing, so e.g.
                # abrupt-4xCO2 + ghg_forcing="external" would silently run the
                # file's (historical) trajectory instead of 4xCO2.
                errors.append(
                    f"experiment {self.experiment!r} prescribes FIXED GHG "
                    "concentrations, but ghg_forcing='external' overrides "
                    "them with the annual file trajectory; use "
                    "ghg_forcing='constant' for fixed-forcing experiments"
                )
        # The adiabatic in-cloud floor reaches radiation through the SHARED
        # physics pipeline (build_physics_pipeline -> build_cloud_config), which
        # serves the cd-grid family (cdgrid + aliases 'centered'/'finite_volume'),
        # latlon_cgrid, and the ML backends — all of which DO thread it.  Only
        # MPAS and spectral build RadiationConfig directly, bypassing the pipeline,
        # and would SILENTLY fall back to the constant floor.  Reject the opt-in on
        # ONLY those genuinely-bypassing backends (dispatch-hardening — a silent
        # no-op is a hard error), so a cd-grid alias is not falsely blocked
        # (codex/review).
        _NO_CLOUD_THREAD_DISCRETIZATIONS = ("mpas", "spectral")
        if (self.cloud_diagnostic_condensate_scheme != "constant"
                and self.dycore.discretization
                in _NO_CLOUD_THREAD_DISCRETIZATIONS):
            errors.append(
                "cloud_diagnostic_condensate_scheme="
                f"{self.cloud_diagnostic_condensate_scheme!r} is not wired into "
                f"the {self.dycore.discretization!r} radiation path (that backend "
                "builds RadiationConfig directly, bypassing the shared cloud "
                "pipeline); it would silently run 'constant'.  Use a "
                "pipeline backend (cd-grid / latlon) or scheme='constant'."
            )
        # use_clubb_cloud_fraction is enforced (turbulence must be clubb) by
        # build_physics_pipeline (cd-grid / latlon) and by combined.make_physics
        # (MPAS); the spectral standalone radiation path builds neither, so the
        # opt-in would silently no-op there.  Reject it loudly on that backend
        # (dispatch-hardening, mirrors the condensate-scheme guard above).
        if (self.use_clubb_cloud_fraction
                and self.dycore.discretization == "spectral"):
            errors.append(
                "use_clubb_cloud_fraction=True is not wired into the "
                f"{self.dycore.discretization!r} radiation path (that backend "
                "builds RadiationConfig directly, bypassing the shared physics "
                "pipeline that enforces it); it would silently no-op.  Use a "
                "pipeline backend (cd-grid / latlon) or MPAS with "
                "turbulence='clubb', or drop --use-clubb-cloud-fraction."
            )
        # The CLUBB cloud-fraction routing needs the producer on EVERY lane
        # (build_physics_pipeline / combined.make_physics both raise at build
        # time; fail at config time too so a deck is refused before setup).
        if self.use_clubb_cloud_fraction and self.turbulence != "clubb":
            errors.append(
                "use_clubb_cloud_fraction=True requires turbulence='clubb' "
                "(the only cloud-fraction-producing closure); got "
                f"{self.turbulence!r}."
            )
        # CAM6 cloud fraction (cldfrc2m ice stratus + CLUBB liquid + deepcu):
        # the liquid part IS the CLUBB PDF fraction, so it needs the producer
        # and the carry routing; wired on the MPAS combined-physics lane only
        # (the FV pipeline's cloud call and the cube CMOR collector do not
        # thread lat / the deep-convection carries).
        if self.cloud_scheme == "cam6_clubb":
            if self.turbulence != "clubb":
                errors.append(
                    "cloud_scheme='cam6_clubb' takes its liquid cloud fraction "
                    "from CLUBB's PDF; requires turbulence='clubb', got "
                    f"{self.turbulence!r}."
                )
            if not self.use_clubb_cloud_fraction:
                errors.append(
                    "cloud_scheme='cam6_clubb' requires "
                    "use_clubb_cloud_fraction=True (routes the CLUBB cloud-"
                    "fraction carry to radiation); got False."
                )
            if self.dycore.discretization != "mpas":
                errors.append(
                    "cloud_scheme='cam6_clubb' is wired on the MPAS lane only; "
                    f"got discretization={self.dycore.discretization!r}."
                )
            if self.convective_cloud:
                errors.append(
                    "cloud_scheme='cam6_clubb' carries its own deep-convective "
                    "fraction (clubb_intr deepcu); convective_cloud=True is "
                    "not part of CAM6 and is refused."
                )
            if self.microphysics == "none":
                errors.append(
                    "cloud_scheme='cam6_clubb' has no diagnostic condensate "
                    "floor (CAM6 radiates the prognostic condensate); requires "
                    "a microphysics scheme, got microphysics='none'."
                )
            if self.radiation not in ("rrtmgp", "rrtmg"):
                errors.append(
                    "cloud_scheme='cam6_clubb' needs a cloud-path radiation "
                    "(rrtmgp/rrtmg) -- 'gray'/'none' return before the cloud "
                    "call, so the CAM6 cloud field would radiate nothing while "
                    f"clt/clivi report it; got radiation={self.radiation!r}."
                )
        # Cross-field: the diagnostic-condensate FLOOR exists only for the
        # sub-grid diagnostic-fraction schemes (sundqvist / xu_randall); 'none'
        # skips clouds and 'resolved' (CRM) excludes the floor.  It is radiatively
        # active only when radiation consumes cloud paths (rrtmgp / rrtmg); 'none'
        # and 'gray' ignore them.  Reject 'adiabatic' in combinations where it
        # would be a silent no-op (dispatch-hardening).
        if self.cloud_diagnostic_condensate_scheme != "constant":
            if self.cloud_scheme not in ("sundqvist", "xu_randall"):
                errors.append(
                    "cloud_diagnostic_condensate_scheme="
                    f"{self.cloud_diagnostic_condensate_scheme!r} only affects the "
                    f"sundqvist/xu_randall diagnostic floor; cloud_scheme="
                    f"{self.cloud_scheme!r} would ignore it."
                )
            if self.radiation not in ("rrtmgp", "rrtmg"):
                errors.append(
                    "cloud_diagnostic_condensate_scheme="
                    f"{self.cloud_diagnostic_condensate_scheme!r} needs a "
                    f"cloud-path radiation (rrtmgp/rrtmg); radiation="
                    f"{self.radiation!r} ignores cloud paths."
                )
        if self.microphysics not in VALID_MICROPHYSICS:
            errors.append(
                f"microphysics must be one of {VALID_MICROPHYSICS}, "
                f"got {self.microphysics!r}"
            )
        # Physics-scheme membership (mirror the integration.py factory sets so
        # a typo fails here, not only at JIT-compile inside integration.py).
        _valid_convection = VALID_CONVECTION_SCHEMES
        if self.convection not in _valid_convection:
            errors.append(
                f"convection must be one of {_valid_convection}, "
                f"got {self.convection!r}"
            )
        _valid_turbulence = VALID_TURBULENCE
        if self.turbulence not in _valid_turbulence:
            errors.append(
                f"turbulence must be one of {_valid_turbulence}, "
                f"got {self.turbulence!r}"
            )
        if (self.clubb_trop_cloud_top_press is not None
                and self.turbulence != "clubb"):
            errors.append(
                "clubb_trop_cloud_top_press is a CLUBB field; got "
                f"turbulence={self.turbulence!r} (also refused with a "
                "turbulence_override, which bypasses the CLUBB branch)"
            )
        if self.clubb_q_flux_scale is not None:
            if self.turbulence != "clubb":
                errors.append(
                    "clubb_q_flux_scale is a CLUBB field; got "
                    f"turbulence={self.turbulence!r}")
            if self.clubb_prognostic:
                errors.append(
                    "clubb_q_flux_scale is a diagnostic-CLUBB mechanism probe; "
                    "the prognostic closure does not read it")
            if self.turbulence_override is not None:
                errors.append(
                    "clubb_q_flux_scale is refused with a turbulence_override "
                    "(the override is authoritative; set CLUBBConfig."
                    "q_flux_scale inside it)")
            band = self.clubb_q_flux_scale_sigma_band
            try:
                lo, hi = (float(band[0]), float(band[1])) if len(band) == 2 else (None, None)
            except (TypeError, ValueError):
                lo, hi = None, None
            if lo is None or not (0.0 <= lo < hi <= 1.0):
                errors.append(
                    "clubb_q_flux_scale needs clubb_q_flux_scale_sigma_band "
                    f"(lo, hi) with 0 <= lo < hi <= 1, got {band!r}")
        elif self.clubb_q_flux_scale_sigma_band is not None:
            errors.append(
                "clubb_q_flux_scale_sigma_band is set but clubb_q_flux_scale is "
                "None: the band does nothing on its own")
        _valid_surface_bulk = VALID_SURFACE_BULK
        if self.surface_bulk_scheme not in _valid_surface_bulk:
            errors.append(
                f"surface_bulk_scheme must be one of {_valid_surface_bulk}, "
                f"got {self.surface_bulk_scheme!r}"
            )
        _valid_thermo_conventions = ("legoesm", "aerobulk")
        if self.surface_thermo_convention not in _valid_thermo_conventions:
            errors.append(
                f"surface_thermo_convention must be one of "
                f"{_valid_thermo_conventions}, "
                f"got {self.surface_thermo_convention!r}"
            )
        _valid_stability = ("dyer1974", "beljaars_holtslag1991",
                            "grachev2007_sheba", "gryanik2020")
        # A tiled surface WITH REAL LAND is exempt from the second check
        # below: its land tile is ocean_cfg._replace(bulk_scheme="most", ...)
        # (physics_pipeline ~:701), so it INHERITS the injected
        # stability_scheme and runs the land Monin-Obukhov law with it — real
        # land fluxes move even though the top-level scheme is "constant"
        # (codex R3 P2: the first form of that guard was too broad and would
        # have rejected this working configuration).  "Real land" is the same
        # predicate used elsewhere in this method: an explicit mask, or an
        # active tile with a topography that actually derives f_land > 0 —
        # an IDEALIZED topography ("flat" or "gaussian") gives f_land == 0 in
        # every cell, so the tiled path never engages and the scheme would be
        # diagnostic-only again (codex R4 P2).
        # IDEALIZED topography derives no land: "flat" is zero elevation and
        # "gaussian" is an idealized bell that the driver deliberately leaves
        # all-ocean (its positive tails would otherwise label the entire globe
        # land).  Only a real elevation file or an explicit mask gives land.
        _idealized_topography = self.topography in ("flat", "gaussian")
        _tiled_with_real_land = self.surface_tiled and bool(
            self.land_mask_path
            or ((self.slab_land_active or self.use_multilayer_land)
                and not _idealized_topography))
        if self.surface_stability_scheme not in _valid_stability:
            errors.append(
                f"surface_stability_scheme must be one of {_valid_stability}, "
                f"got {self.surface_stability_scheme!r}"
            )
        # The stability selector only acts on the STABLE branch of the
        # iterative MOST solver, and surface_layer.compute_surface_fluxes
        # routes ONLY ("most", "coare3", "large_yeager") through that solver —
        # bulk_scheme="constant" takes the constant-Cd path and ignores
        # stability entirely.  But the 2 m diagnostic (diagnostics.py) builds
        # a COARE profile with the selected scheme regardless, so the pair
        # would leave the physics untouched while MOVING the published tas.
        # A knob that changes only the diagnostic is worse than an inert one
        # (codex review P2) — reject it, unless _tiled_with_real_land above.
        elif (self.surface_stability_scheme != "dyer1974"
                and self.surface_bulk_scheme == "constant"
                and not _tiled_with_real_land):
            errors.append(
                f"surface_stability_scheme="
                f"{self.surface_stability_scheme!r} needs a "
                f"stability-dependent surface_bulk_scheme ('coare3' or "
                f"'large_yeager' — 'most' is a turbulence-config bulk_scheme, "
                f"not an experiment-level one, and VALID_SURFACE_BULK "
                f"rejects it), or surface_tiled=True whose land tile runs "
                f"MOST; with 'constant' and untiled surfaces the fluxes "
                f"ignore the stable branch entirely and only the 2 m "
                f"diagnostic would move."
            )
        # A non-"constant" surface scheme upgrades the ATMOSPHERE surface layer
        # (via _resolve_turbulence on the turbulence config).  With
        # turbulence="none" there is no SurfaceLayerConfig to update, so the
        # atmosphere would silently stay on its fallback fluxes while the
        # ocean/coupler tiles switch to MOST — an inconsistent interface.  Reject
        # loudly (codex review HIGH#2).
        if self.surface_bulk_scheme != "constant" and self.turbulence == "none":
            errors.append(
                f"surface_bulk_scheme={self.surface_bulk_scheme!r} requires a "
                f"turbulence scheme (turbulence != 'none') so the atmosphere "
                f"surface layer uses the same bulk-flux algorithm as the ocean "
                f"tile; got turbulence='none'."
            )
        # Tiled (mosaic) surface fluxes inject a per-tile flux as the BL bottom
        # boundary condition; only the kernels that accept the injected
        # ``surface_flux=(tau_x, tau_y, shflx, lhflx, ustar)`` tuple can consume
        # it (louis and the CLUBB family — clubb_lite / clubb, the latter routing
        # the flux through clubb_step's kinematic prescribed-BC interface).
        # Reject any other scheme loudly rather than silently ignoring the
        # request (dispatch hardening).
        _tiled_turbulence = ("louis", "clubb_lite", "clubb")
        if self.surface_tiled and self.turbulence not in _tiled_turbulence:
            errors.append(
                f"surface_tiled=True is currently supported only with "
                f"turbulence in {_tiled_turbulence} (the kernels that consume the "
                f"injected tiled surface flux); got turbulence={self.turbulence!r}."
            )
        if self.surface_tiled and not (self.slab_land_active
                                       or self.land_mask_path
                                       or self.use_multilayer_land):
            errors.append(
                "surface_tiled=True requires an active land tile "
                "(--slab-land-active, --use-multilayer-land, or a land-mask file); "
                "otherwise there is no land tile to give its own surface scheme. "
                "use_multilayer_land is an active tile whose land fraction comes "
                "from --topography (elevation-derived f_land) when no mask is given."
            )
        # ...but the multilayer tile can only get f_land from topography — a FLAT
        # topography gives f_land==0 everywhere (no land), which would silently
        # no-op the requested land tile.  Require real topography OR an explicit
        # mask when multilayer is the sole land-tile signal.
        # Any requested land tile, slab or multilayer, tiled or not: an
        # idealized topography derives no land at all, so the tile runs over
        # zero land and goes silently inert -- the trap that cost the AMIP
        # campaign three tuning waves. Refused, not warned.
        if ((self.use_multilayer_land or self.slab_land_active)
                and not self.land_mask_path
                and _idealized_topography):
            errors.append(
                f"a land tile was requested (use_multilayer_land="
                f"{self.use_multilayer_land}, slab_land_active="
                f"{self.slab_land_active}) with an idealized "
                f"topography={self.topography!r} and no land-mask file, which "
                "has NO land anywhere (f_land is 0 in every cell), so the tile "
                "would silently no-op — pass a real elevation file to --topography, or a "
                "--land-mask-file."
            )
        # The same "no land anywhere" trap, on the MESH lane, which does not use
        # surface_tiled and so never reached the check above: the flux handoff
        # (and any calibrated canopy conductance riding it) would build no soil
        # column at all and go silently inert.  A degenerate all-ocean MASK FILE
        # cannot be seen from the config.  The driver checks that case too, but
        # only fatally for a SINGLE-rank run: under a cell partition a rank
        # legitimately owns no land, and the cross-rank vote that would settle it
        # cannot be placed safely (it would sit downstream of per-rank setup that
        # can raise, leaving peers blocked).  Counting land points in the mask
        # file HERE, where the check is rank-symmetric, is the open follow-up.
        if (self.mpas_land_beta_soil and not self.land_mask_path
                and _idealized_topography):
            errors.append(
                f"mpas_land_beta_soil with an idealized topography"
                f"={self.topography!r} and no land-mask file has NO land "
                "(f_land is 0 everywhere), so no soil column is built and the "
                "land-flux handoff is silently inert — pass a real "
                "--topography or a --land-mask-file."
            )
        # Deploying the baked land parameters under the physics they were
        # calibrated under — the biophysics LMIP two-leaf canopy. The
        # overlapping keys are CHECKED, not overridden, so a run can never
        # believe it is on the calibrated model while one key disagrees; the
        # settings with no config key of their own (soil growth factor, the
        # canopy's intrinsic stomata) come from the one shared setup.
        if self.land_calibrated_physics:
            from legoesm.land.config import biophysics_lmip_two_leaf_setup
            _cal = biophysics_lmip_two_leaf_setup()
            _grid = _cal["soil_grid"]
            if not self.use_multilayer_land:
                errors.append(
                    "land_calibrated_physics=True requires use_multilayer_land=True: "
                    "the calibrated tables are a MULTILAYER bake and the slab tile "
                    "has no equivalent (its coupled implementation is the driver's "
                    "own _step_slab_land, not the calibrated legoesm.land.slab_land "
                    "model)."
                )
            _want = {
                # The canopy runs its own Ball-Berry stomata, so the coupled
                # stomatal beta must be OFF: on, it throttles the same
                # conductance a second time.
                "land_stomatal_beta": (self.land_stomatal_beta, False),
                "land_surface_scheme": (self.land_surface_scheme, "two_leaf"),
                "multilayer_n_layers": (self.multilayer_n_layers, _grid.n_layers),
                "multilayer_soil_depth": (self.multilayer_soil_depth,
                                          _grid.total_depth),
                # gs_max only feeds the Jarvis model, which the calibrated setup
                # does not select — but the setup replaces the whole stomatal
                # config, so a value set here would vanish without a word.
                "land_gs_max": (self.land_gs_max, _cal["stomata"].gs_max),
                # The land setup forces this on, so `false` here would pass
                # validation and then be silently reversed.
                "snow_albedo_feedback": (self.snow_albedo_feedback,
                                         _cal["snow_albedo_feedback"]),
            }
            for _key, (_got, _exp) in _want.items():
                if _got != _exp:
                    errors.append(
                        f"land_calibrated_physics=True requires {_key}={_exp!r} "
                        f"(the value the baked tables were fitted under); got "
                        f"{_got!r}. Set it or drop land_calibrated_physics — "
                        "these are not silently overridden."
                    )
            # ONLY THE UNSTRUCTURED (MESH) LANE CAN CARRY THIS TODAY, and the
            # reason is specific: that lane hands the atmosphere the land tile's
            # SOLVED sensible and latent fluxes, so the calibrated canopy
            # conductance arrives as the number the land model computed.  Every
            # other lane re-derives the land flux from a scaled saturation
            # humidity, which cannot reproduce it — the land scheme applies its
            # throttle to the humidity GRADIENT and bypasses it for snow and
            # dew, so the re-derivation can reach the opposite SIGN, and the
            # atmosphere's exchange coefficient uses a scalar roughness where
            # the land uses a per-column tuned one.  Both reviewers refuted the
            # humidity route independently (2026-08-19).  Extending the FLUX
            # handoff to the structured lanes means carrying those fluxes at the
            # radiation cadence through SegmentCarry; until then, refuse rather
            # than deploy the tables under a coupling that cannot express them.
            _is_mesh_lane = (d.discretization == "mpas"
                             or normalize_grid_type(g.grid_type) == "mpas")
            if not _is_mesh_lane:
                errors.append(
                    "land_calibrated_physics=True is supported only on the MPAS "
                    f"(Voronoi) lane today; discretization={d.discretization!r} "
                    "re-derives the land flux from a scaled saturation humidity "
                    "instead of taking the land tile's solved flux, which cannot "
                    "reproduce the calibrated canopy conductance (and can reach "
                    "the wrong sign over snow and dew). Run the calibrated arm on "
                    "the mesh lane. (There is no partial form: the soil growth "
                    "factor and the carbon scheme this switch supplies have no "
                    "config keys of their own.)"
                )
        # Transient land-use cover re-weights the MULTILAYER land vegetation params
        # from a transient surfdata; without both signals it would silently no-op
        # (there is no slab-land transient-cover path).  Fail early rather than run
        # a static-cover land while the user believes cover is evolving.
        if self.transient_land_cover:
            if not self.use_multilayer_land:
                errors.append(
                    "transient_land_cover=True requires use_multilayer_land=True: "
                    "the transient cover re-weights the multilayer LandSurfaceParams "
                    "(there is no slab-land transient-cover path)."
                )
            if not self.land_cover_surfdata:
                errors.append(
                    "transient_land_cover=True requires land_cover_surfdata to point "
                    "at a transient legoesm_surfdata NetCDF (pft_frac(year, npft, "
                    "lat, lon) in percent, CLM5 17-PFT axis); none was set."
                )
        if not (self.surface_z0_land > 0.0):
            errors.append(
                f"surface_z0_land must be a positive roughness length [m]; "
                f"got {self.surface_z0_land!r}."
            )
        # Prognostic soil-water bucket: needs an active land tile to step,
        # and physically-bounded parameters.
        if self.land_soil_bucket and not (
                self.slab_land_active or self.land_mask_path):
            errors.append(
                "land_soil_bucket=True requires an active land tile "
                "(--slab-land-active or a land-mask file); there is no land "
                "surface to carry soil water otherwise."
            )
        # NaN-safe guards: ``NaN > 0`` / ``NaN < 0`` are both False, so a bare
        # comparison would let a NaN slip through and poison the infiltration cap.
        if not (math.isfinite(self.land_K_infiltration)
                and self.land_K_infiltration > 0.0):
            errors.append(
                f"land_K_infiltration must be a positive, finite saturated "
                f"infiltration capacity [m/s]; got {self.land_K_infiltration!r}."
            )
        if not (math.isfinite(self.land_infil_suction_boost)
                and self.land_infil_suction_boost >= 0.0):
            errors.append(
                f"land_infil_suction_boost (Green-Ampt psi_f/L_f) must be finite "
                f"and >= 0; got {self.land_infil_suction_boost!r}."
            )
        if not (self.land_bucket_w_max > 0.0):
            errors.append(
                f"land_bucket_w_max must be a positive bucket capacity "
                f"[kg/m^2]; got {self.land_bucket_w_max!r}."
            )
        if not (0.0 < self.land_beta_min <= 1.0):
            errors.append(
                f"land_beta_min (min soil-moisture availability) must be in "
                f"(0, 1]; got {self.land_beta_min!r}."
            )
        if not (0.0 <= self.land_bucket_w_init_frac <= 1.0):
            errors.append(
                f"land_bucket_w_init_frac (initial fill fraction) must be in "
                f"[0, 1]; got {self.land_bucket_w_init_frac!r}."
            )
        # --- MPAS lane land-surface consistency (fail-fast, no silent no-ops)
        # The standalone MPAS lane builds its physics from combined.make_physics
        # (NOT the driver PhysicsPipeline), so the pipeline land tile
        # (slab_land_active / land_soil_bucket / surface_tiled) never executes
        # there — accepting those flags on MPAS ran a 60-day A/B against a
        # byte-identical twin (2026-07-23).  Conversely the MPAS land boundary
        # knobs are consumed only by the MPAS lane.
        # Lane detection keys on BOTH fields: the _run_mpas dispatch actually
        # keys on grid_type (any Voronoi alias), and discretization stays
        # consistent only via run_amip's postprocessor; a mismatched pair is
        # fail-closed at the component factory, but the guard here must not
        # emit a wrong-lane message for grid_type-keyed configs (codex F4,
        # alias set via normalize_grid_type per codex F-B3).
        _is_mpas = (d.discretization == "mpas"
                    or normalize_grid_type(g.grid_type) == "mpas")
        _pus = self.physics_update_steps
        if not isinstance(_pus, int) or isinstance(_pus, bool) or _pus < 1:
            errors.append(
                f"physics_update_steps must be an int >= 1, got {_pus!r}")
        elif _pus > 1:
            if not _is_mpas:
                errors.append(
                    f"physics_update_steps={_pus} is wired into the MPAS lane "
                    "only (combined.make_physics cadence variants); it would "
                    "be silently inert here")
            _rus = self.rad_update_steps
            if _rus < _pus or _rus % _pus != 0:
                errors.append(
                    f"rad_update_steps={_rus} must be a multiple of "
                    f"physics_update_steps={_pus} (>= it) so radiation is "
                    "recomputed on physics steps; a held-physics step cannot "
                    "solve radiation")
        _nmm = self.cld_macmic_num_steps
        if not isinstance(_nmm, int) or isinstance(_nmm, bool) or _nmm < 1:
            errors.append(
                f"cld_macmic_num_steps must be an int >= 1, got {_nmm!r}")
        elif _nmm > 1:
            if not _is_mpas:
                errors.append(
                    f"cld_macmic_num_steps={_nmm} is wired into the MPAS lane "
                    "only (combined.make_physics macmic loop); it would be "
                    "silently inert here")
            if self.turbulence == "none" and self.microphysics == "none":
                errors.append(
                    f"cld_macmic_num_steps={_nmm} sub-cycles turbulence and "
                    "microphysics, but both are 'none'")
        if self._liquid_partition_resolved():
            # CLUBB's liquid exchange REPLACES the host's cloud water with the
            # closure's equilibrium diagnosis, so the microphysics must read the
            # REPLACED value; CAM guarantees that by sequential-update splitting
            # inside its macmic loop (physpkg.F90:2097-2101).  This model uses
            # that order only when the loop runs: at cld_macmic_num_steps=1 the
            # combined physics takes the PARALLEL branch
            # (combined.py ``_accumulate_step``), where every module is
            # evaluated on the same start-of-step state and the tendencies are
            # SUMMED.  The final liquid would then be the closure's equilibrium
            # PLUS a microphysical increment computed from the stale, 2-3x
            # smaller liquid -- and since autoconversion goes as roughly the
            # 2.5th power of cloud water, that sink is wrong by nearly an order
            # of magnitude.  Water is still conserved, so nothing would fail
            # loudly; the climate would simply be wrong.  Refuse instead.
            if _nmm is None or not isinstance(_nmm, int) or _nmm < 2:
                errors.append(
                    "clubb_liquid_partition=True needs cld_macmic_num_steps>=2: "
                    "the closure REPLACES the host cloud water, so the "
                    "microphysics has to run on the replaced value, and only "
                    "the macro/micro sub-cycle applies the modules in sequence. "
                    f"At cld_macmic_num_steps={_nmm!r} they are evaluated in "
                    "parallel on the same state and summed, which leaves the "
                    "microphysical sinks evaluated on the pre-exchange liquid")
            # The diagnostic condensate floors exist to compensate for the very
            # liquid this lever restores, so with it on they are added on top of
            # the closure's own water and the clouds are opaque twice over.
            # They act on the RADIATIVE condensate, not on the prognostic
            # tracers, so this is an opacity error rather than a break in the
            # exchange's tracer-water pairing (codex corrected the rationale) --
            # but it is still a mechanism to switch OFF, not a number to retune.
            #
            # Checked against the RESOLVED cloud config, because ``None`` here
            # does not mean "no floor": it means "take the scheme's default",
            # which is non-zero.  Both floors count: the stratiform
            # ``q_c_diagnostic`` and the independent convective
            # ``conv_cloud_condensate`` term in cloud_fraction.py.
            try:
                from legoesm.atmosphere.physics.clouds.config import (
                    build_cloud_config,
                )
                _cc = build_cloud_config(
                    self.cloud_scheme,
                    q_c_diagnostic=self.cloud_q_c_diagnostic,
                    conv_cloud_condensate=self.cloud_conv_cloud_condensate,
                    convective_cloud=bool(self.convective_cloud))
            except Exception:                       # pragma: no cover
                _cc = None                          # reported by its own check
            if _cc is not None:
                _floors = [
                    (n, v) for n, v in
                    (("q_c_diagnostic", getattr(_cc, "q_c_diagnostic", 0.0)),
                     ("conv_cloud_condensate",
                      getattr(_cc, "conv_cloud_condensate", 0.0)
                      if bool(self.convective_cloud) else 0.0))
                    if v
                ]
                if _floors:
                    errors.append(
                        "clubb_liquid_partition=True leaves a diagnostic "
                        "condensate floor active in the RESOLVED cloud config "
                        f"({', '.join(f'{n}={v!r}' for n, v in _floors)}). "
                        "Those floors exist to compensate for the missing "
                        "closure liquid this lever restores, so they would be "
                        "imposed on top of it and the clouds would be made "
                        "opaque twice. Set them to 0 explicitly (None means "
                        "the scheme's non-zero default, not 'off')")
        if _is_mpas:
            for _flag in ("slab_land_active", "land_soil_bucket",
                          "surface_tiled"):
                if getattr(self, _flag):
                    errors.append(
                        f"{_flag}=True is silently inert on the MPAS lane "
                        "(its physics comes from combined.make_physics, not "
                        "the driver pipeline). Use the MPAS land boundary "
                        "knobs instead: mpas_land_lapse_K_per_km / "
                        "mpas_land_beta."
                    )
            # Inert-corner rejection (codex F1-F3): each knob needs the
            # machinery it modifies to actually be on.
            if (self.mpas_land_lapse_K_per_km > 0.0
                    and self.radiation == "none"):
                errors.append(
                    "mpas_land_lapse_K_per_km adjusts the SST-forcing "
                    "surface anchor, which is only built when radiation != "
                    "'none' — the knob would be silently inert."
                )
            if self.mpas_land_beta != 1.0 and self.turbulence == "none":
                errors.append(
                    "mpas_land_beta throttles the turbulence surface "
                    "humidity; turbulence='none' has no surface latent flux "
                    "to throttle — the knob would be silently inert."
                )
            if (self.land_calibrated_physics and self.use_multilayer_land
                    and not self.mpas_land_beta_soil):
                # Without this the mesh lane SOLVES the land tile's humidity
                # and fluxes and then throws them away, keeping the static
                # mpas_land_beta instead — so the fitted plant model would
                # change the land tile's own temperature and nothing the
                # atmosphere sees (codex round 3).  A calibrated run whose
                # canopy conductance never reaches the atmosphere is the same
                # inert-parameter defect the flag exists to remove.
                errors.append(
                    "land_calibrated_physics=True on the MPAS lane requires "
                    "mpas_land_beta_soil=True: without it the lane discards the "
                    "land tile's solved humidity and fluxes and keeps the static "
                    "mpas_land_beta, so the calibrated canopy conductance would "
                    "never reach the atmosphere."
                )
            if self.mpas_land_beta_soil and self.turbulence != "none":
                # The land tile's SOLVED fluxes are handed to the turbulence
                # kernel, and a kernel whose signature has no ``surface_flux``
                # argument REFUSES them — at run time, after the job has started.
                # Ask the same signature scan the runtime guard uses, so a run
                # that could never couple the land fluxes fails at config time
                # instead of hours in (codex round 5).
                from legoesm.atmosphere.physics.turbulence.integration import (
                    schemes_accepting_surface_flux,
                )
                _flux_ok = schemes_accepting_surface_flux()
                if self.turbulence not in _flux_ok:
                    errors.append(
                        f"mpas_land_beta_soil=True hands the land tile's solved "
                        f"surface fluxes to the turbulence scheme, but "
                        f"{self.turbulence!r} takes no 'surface_flux' argument "
                        f"and refuses them at run time. Use one of {_flux_ok}."
                    )
            if self.mpas_land_beta_soil:
                # Traced beta_soil needs the multilayer land producing it and
                # the turbulence surface flux consuming it (inert-corner
                # rejection, same doctrine as the static knobs above).
                if not self.use_multilayer_land:
                    errors.append(
                        "mpas_land_beta_soil threads the multilayer land's "
                        "root-zone beta_soil into the turbulence surface "
                        "humidity; it requires use_multilayer_land=True."
                    )
                if self.turbulence == "none":
                    errors.append(
                        "mpas_land_beta_soil throttles the turbulence "
                        "surface humidity; turbulence='none' has no surface "
                        "latent flux to throttle — the flag would be "
                        "silently inert."
                    )
            # flat topography yields all-zero f_land UNLESS an explicit land
            # mask overrides it (codex F-B2); the driver's runtime all-zero
            # guard remains authoritative for degenerate mask files.
            if ((self.mpas_land_lapse_K_per_km > 0.0
                 or self.mpas_land_beta != 1.0)
                    and self.topography in ("flat", "gaussian")
                    and not self.land_mask_path):
                errors.append(
                    "mpas_land_lapse_K_per_km/mpas_land_beta need a land "
                    f"fraction, but topography={self.topography!r} (with no land-mask "
                    "file) yields an all-zero f_land — the knobs would "
                    "change nothing."
                )
        else:
            if self.mpas_land_lapse_K_per_km != 0.0 or self.mpas_land_beta != 1.0:
                errors.append(
                    "mpas_land_lapse_K_per_km/mpas_land_beta are MPAS-lane "
                    f"knobs; discretization={d.discretization!r} has its own "
                    "land tile (slab_land_active / use_multilayer_land) and "
                    "would silently ignore them."
                )
            if self.mpas_land_beta_soil:
                errors.append(
                    "mpas_land_beta_soil is an MPAS-lane flag; "
                    f"discretization={d.discretization!r} threads beta_soil "
                    "through its own tiled land pipeline "
                    "(physics_pipeline._land_tile_q_sfc) and would silently "
                    "ignore it."
                )
            if self.mpas_qv_smooth_del2_m2s != 0.0:
                errors.append(
                    "mpas_qv_smooth_del2_m2s is an MPAS-lane knob; "
                    f"discretization={d.discretization!r} already smooths q_v "
                    "in its step factories (qv_smooth_coeff) and would "
                    "silently ignore it."
                )
            if self.mpas_qv_smooth_del4_m4s != 0.0:
                errors.append(
                    "mpas_qv_smooth_del4_m4s is an MPAS-lane knob; "
                    f"discretization={d.discretization!r} already smooths q_v "
                    "in its step factories (qv_smooth_coeff) and would "
                    "silently ignore it."
                )
            if self.mpas_ice_skin_prognostic:
                errors.append(
                    "mpas_ice_skin_prognostic is an MPAS-lane knob; "
                    f"discretization={d.discretization!r} has its own "
                    "surface/ice tiles and would silently ignore it."
                )
        if not (math.isfinite(self.mpas_land_lapse_K_per_km)
                and 0.0 <= self.mpas_land_lapse_K_per_km <= 20.0):
            errors.append(
                f"mpas_land_lapse_K_per_km must be finite in [0, 20] "
                f"(0=off, 6.5=ICAO standard); got "
                f"{self.mpas_land_lapse_K_per_km!r}."
            )
        if not (math.isfinite(self.mpas_land_beta)
                and 0.0 <= self.mpas_land_beta <= 1.0):
            errors.append(
                f"mpas_land_beta (land evaporation efficiency) must be finite "
                f"in [0, 1]; got {self.mpas_land_beta!r}."
            )
        # Numerics diffusivity, not a trainable closure: 0 = off; upper bound
        # 1e8 m^2/s is far above any del2 a stable explicit step admits (the
        # driver additionally enforces the mesh-specific CFL/monotonicity
        # bound at setup — presence of a q_v tracer is checked there too).
        if not (math.isfinite(self.mpas_qv_smooth_del2_m2s)
                and 0.0 <= self.mpas_qv_smooth_del2_m2s <= 1.0e8):
            errors.append(
                f"mpas_qv_smooth_del2_m2s (horizontal q_v del2 diffusivity "
                f"[m^2/s]) must be finite in [0, 1e8]; got "
                f"{self.mpas_qv_smooth_del2_m2s!r}."
            )
        # Same shape one order up: 0 = off, and the ceiling is far above any
        # biharmonic a stable explicit step admits (the driver enforces the
        # mesh-specific bound nu4*dt*g_max^2 <= 0.5 at setup).
        if not (math.isfinite(self.mpas_qv_smooth_del4_m4s)
                and 0.0 <= self.mpas_qv_smooth_del4_m4s <= 1.0e18):
            errors.append(
                f"mpas_qv_smooth_del4_m4s (horizontal q_v del4 diffusivity "
                f"[m^4/s]) must be finite in [0, 1e18]; got "
                f"{self.mpas_qv_smooth_del4_m4s!r}."
            )
        # Ice-skin inert corners: the skin integrates the physics' exported
        # surface fluxes (radiation channel), so radiation="none" would leave
        # it frozen at its seed forever; a thickness override without the
        # boolean gate would be silently inert.
        # "rrtmg" is this driver's alias for the RRTMGP builder (the production
        # deck spells it that way); both reach the standalone radiation path.
        if self.cloud_cap_floor_on and (self.cloud_scheme == "none"
                                        or self.radiation not in ("rrtmgp", "rrtmg")):
            errors.append(
                "cloud_cap_floor_on requires an active cloud scheme and rrtmgp/rrtmg "
                f"radiation (cloud_scheme={self.cloud_scheme!r}, radiation={self.radiation!r}); "
                "otherwise the polar-cap radiative floor would be a silent no-op.")
        if self.mpas_ice_skin_prognostic and self.radiation == "none":
            errors.append(
                "mpas_ice_skin_prognostic integrates the surface energy "
                "fluxes exported by the physics; radiation='none' computes "
                "none — the skin would stay at its seed forever."
            )
        if (self.mpas_ice_thickness_m != 2.0
                and not self.mpas_ice_skin_prognostic):
            errors.append(
                f"mpas_ice_thickness_m={self.mpas_ice_thickness_m!r} requires "
                "mpas_ice_skin_prognostic=True (the override would be "
                "silently inert)."
            )
        if not (math.isfinite(self.mpas_ice_thickness_m)
                and 0.1 <= self.mpas_ice_thickness_m <= 10.0):
            errors.append(
                f"mpas_ice_thickness_m (climatological ice slab [m]) must be "
                f"finite in [0.1, 10]; got {self.mpas_ice_thickness_m!r}."
            )
        # Ice-curve drain preconditions (fail-fast: every one is a silent no-op
        # or an unphysical deposition otherwise).  It is a Morrison-on-MPAS
        # post-step correction that gates+lands on the blended liquid/ice curve
        # and DEPOSITS to prognostic cloud ice q_i (seeding ice number N_i):
        #   (a) needs the boolean drain gate it modifies (else inert);
        #   (b) is only wired into the MPAS post-step hook (else inert on other
        #       lanes — the in-scheme liquid adjustment is untouched);
        #   (c) needs Morrison's prognostic q_i + N_i + nucleation mass (other
        #       schemes lack a cloud-ice reservoir, so the drain would deposit
        #       LIQUID at ice saturation — re-evaporating, energy-inconsistent).
        if self.hard_sat_ice_curve:
            if not self.hard_saturation_adjustment:
                errors.append(
                    "hard_sat_ice_curve=True requires "
                    "hard_saturation_adjustment=True (the mixed-phase curve "
                    "only modifies the active drain — it would be silently "
                    "inert)."
                )
            if normalize_grid_type(self.grid.grid_type) != "mpas":
                errors.append(
                    "hard_sat_ice_curve=True requires an MPAS grid (the "
                    "ice-curve drain is only wired into the MPAS post-step "
                    "hook; on grid_type="
                    f"{self.grid.grid_type!r} it would be silently inert)."
                )
            if self.microphysics != "morrison":
                errors.append(
                    "hard_sat_ice_curve=True requires microphysics='morrison' "
                    "(the drain deposits to prognostic cloud ice q_i and seeds "
                    "ice number N_i from Morrison's nucleation mass; "
                    f"microphysics={self.microphysics!r} has no such reservoir)."
                )
        # Homogeneous (Koop/Ren-MacKenzie) cirrus nucleation is implemented ONLY
        # in MorrisonConfig (no other scheme carries the field), and
        # microphysics="none" early-returns in _resolve_microphysics / skips the
        # MPAS applier entirely, so the flag would be silently inert there.
        # Reject at CONFIG time rather than deep in the applier (same contract as
        # hard_sat_ice_curve above and hard_saturation_adjustment below).
        if (self.homogeneous_ice_nucleation
                and self.microphysics != "morrison"):
            errors.append(
                "homogeneous_ice_nucleation=True requires "
                "microphysics='morrison' (only MorrisonConfig implements the "
                "Koop/Ren-MacKenzie cirrus nucleation); got microphysics="
                f"{self.microphysics!r}."
            )
        # --- hard-saturation-adjustment overrides (fail-fast, no silent no-op)
        # The float overrides only act when the boolean gate is on; bounds
        # mirror the warm-rain schemes' __param_spec__ ((1, 2) trigger,
        # (0.5, 50) K heating cap).
        for _hs_name, _hs_val, _hs_lo, _hs_hi in (
                ("hard_sat_adjust_threshold",
                 self.hard_sat_adjust_threshold, 1.0, 2.0),
                ("hard_sat_max_heating_K",
                 self.hard_sat_max_heating_K, 0.5, 50.0)):
            if _hs_val is None:
                continue
            if not self.hard_saturation_adjustment:
                errors.append(
                    f"{_hs_name}={_hs_val!r} requires "
                    "hard_saturation_adjustment=True (the override would be "
                    "silently inert). Add --hard-saturation-adjustment or "
                    "drop the override."
                )
            if not (math.isfinite(_hs_val) and _hs_lo <= _hs_val <= _hs_hi):
                errors.append(
                    f"{_hs_name} must be finite in [{_hs_lo}, {_hs_hi}] "
                    f"(scheme __param_spec__ bounds); got {_hs_val!r}."
                )
        # The activation gate (and thus any override) is silently inert unless
        # the selected microphysics is a warm-rain scheme carrying the field:
        # microphysics="none" early-returns (None, None) in _resolve_microphysics
        # / the MPAS post-step drain before the flag is ever read, so the gate
        # would do nothing.  Reject it at config time (codex F3) rather than let
        # it silently no-op.  Same scheme set as the fail-loud runtime raise in
        # microphysics/config.apply_microphysics_experiment_flags.
        # Single source of truth (never re-listed here): the schemes that carry
        # the guard live in microphysics/config.HARD_SAT_GUARD_SCHEMES, with the
        # verified exemption reasons in HARD_SAT_GUARD_EXEMPT.  A hardcoded copy
        # of the tuple silently drifted out of date when sundqvist/ml_emulator
        # gained the guard.
        from legoesm.atmosphere.physics.microphysics.config import (
            HARD_SAT_GUARD_EXEMPT, HARD_SAT_GUARD_SCHEMES,
        )
        if (self.hard_saturation_adjustment
                and self.microphysics not in HARD_SAT_GUARD_SCHEMES):
            _why = HARD_SAT_GUARD_EXEMPT.get(self.microphysics, "")
            errors.append(
                "hard_saturation_adjustment requires a microphysics scheme "
                f"carrying the guard {HARD_SAT_GUARD_SCHEMES}; got "
                f"microphysics={self.microphysics!r} (the guard would be "
                f"silently inert). {_why}".rstrip()
            )
        # gs_max is a physical conductance [mol/m2/s]: must be finite and
        # strictly positive (nan/<=0 would zero or NaN the whole land latent
        # flux).  Upper sanity bound 2.0 is well above the StomataConfig
        # __param_spec__ tunable range (0.099, 0.9).
        if not (0.0 < self.land_gs_max <= 2.0):
            errors.append(
                f"land_gs_max (max stomatal conductance [mol/m2/s]) must be "
                f"finite and in (0, 2]; got {self.land_gs_max!r}."
            )
        # Marine-Sc cloud-top entrainment efficiency: finite, in [0, 1] (matches
        # LouisConfig.__param_spec__; the not(lo<=x<=hi) form also rejects NaN/Inf).
        if not (0.0 <= self.louis_cloudtop_entrainment_efficiency <= 1.0):
            errors.append(
                f"louis_cloudtop_entrainment_efficiency (marine-Sc cloud-top "
                f"entrainment A) must be finite and in [0, 1]; got "
                f"{self.louis_cloudtop_entrainment_efficiency!r}."
            )
        # k_h_scale: negative is anti-diffusive (rejected); wired ONLY into
        # the MPAS thermal diffusion — refuse elsewhere rather than run inert.
        if d.k_h_scale is not None:
            if not isinstance(d.k_h_scale, (int, float)) or not (
                    0.0 <= float(d.k_h_scale) <= 100.0):
                errors.append(
                    f"dycore.k_h_scale must be None or finite in [0, 100]; "
                    f"got {d.k_h_scale!r}.")
            if d.discretization != "mpas":
                errors.append(
                    "dycore.k_h_scale is only wired into the MPAS thermal "
                    "diffusion path; on discretization="
                    f"{d.discretization!r} it would be silently inert. "
                    "Unset it or use the MPAS lane.")
        # Vertical advection scheme: membership first, then the same
        # silently-inert refusal as k_h_scale (MPAS + sigma only).
        _vert_adv_options = ("upwind", "van_leer", "sb")
        _spl, _spf = d.mpas_sponge_del2_top_layers, d.mpas_sponge_del2_top_factor
        if isinstance(_spl, bool) or not isinstance(_spl, int) or _spl < 0 or _spl >= g.nlev:
            errors.append(
                f"dycore.mpas_sponge_del2_top_layers must be an int in [0, grid.nlev) (got {_spl!r})")
        if not (isinstance(_spf, (int, float)) and not isinstance(_spf, bool)
                and math.isfinite(_spf) and _spf >= 1.0):
            errors.append(
                f"dycore.mpas_sponge_del2_top_factor must be a finite number >= 1.0 (got {_spf!r})")
        if (isinstance(_spl, int) and _spl > 0
                and (d.discretization != "mpas" or d.model_type != "hydrostatic")):
            errors.append(
                "dycore.mpas_sponge_del2_top_layers is wired into the hydrostatic MPAS dycore "
                f"only; on discretization={d.discretization!r}/model_type={d.model_type!r} "
                "it would be silently inert")
        if (isinstance(_spl, int) and _spl > 0
                and isinstance(d.a_h_scale, (int, float)) and d.a_h_scale <= 0):
            errors.append(
                "dycore.mpas_sponge_del2_top_layers > 0 is silently inert when "
                f"a_h_scale={d.a_h_scale!r} turns the del2 viscosity off; use a_h_scale > 0 "
                "or set mpas_sponge_del2_top_layers=0")
        _dd4s = getattr(d, "mpas_div_damp4_scale", 0.0)
        if (isinstance(_dd4s, (int, float)) and not isinstance(_dd4s, bool)
                and _dd4s > 0
                and (d.discretization != "mpas" or d.model_type != "hydrostatic")):
            errors.append(
                "dycore.mpas_div_damp4_scale is wired into the hydrostatic MPAS dycore "
                f"only; on discretization={d.discretization!r}/model_type={d.model_type!r} "
                "it would be silently inert")
        if d.mpas_vert_advection_scheme not in _vert_adv_options:
            errors.append(
                f"dycore.mpas_vert_advection_scheme must be one of "
                f"{_vert_adv_options}, got {d.mpas_vert_advection_scheme!r}")
        elif d.mpas_vert_advection_scheme != "upwind":
            if d.discretization != "mpas":
                errors.append(
                    "dycore.mpas_vert_advection_scheme is only wired into the "
                    "MPAS dycore; on discretization="
                    f"{d.discretization!r} it would be silently inert. "
                    "Leave it at 'upwind' or use the MPAS lane.")
            # 'van_leer' is sigma-only; 'sb' (conservative Simmons-Burridge
            # flux form) is HYBRID-only.  Each lane refuses the other's scheme
            # rather than run it silently inert.
            _sb = d.mpas_vert_advection_scheme == "sb"
            if _sb and g.vertical_coord == "sigma":
                errors.append(
                    "dycore.mpas_vert_advection_scheme='sb' is the conservative "
                    "flux form of the HYBRID vertical transport; on "
                    "vertical_coord='sigma' it would be silently inert. "
                    "Use a hybrid vertical_coord, or 'upwind'/'van_leer'.")
            if not _sb and g.vertical_coord != "sigma":
                errors.append(
                    "dycore.mpas_vert_advection_scheme is implemented for the "
                    "sigma vertical coordinate only; on vertical_coord="
                    f"{g.vertical_coord!r} it would be silently inert. "
                    "Leave it at 'upwind' or use vertical_coord='sigma'.")
            # The van-Leer stencil is 4 cells wide.  Reject here rather than
            # deep inside the traced kernel (codex round 3).
            if g.nlev < 4:
                errors.append(
                    f"dycore.mpas_vert_advection_scheme="
                    f"{d.mpas_vert_advection_scheme!r} needs at least 4 "
                    f"vertical levels for its 4-cell stencil; grid.nlev="
                    f"{g.nlev}.")
        # Free-atmosphere diffusivity-floor override: None, or finite in
        # (0, 10] m^2/s (the not(lo<x<=hi) form also rejects NaN/Inf).
        if self.hb_kvf_min is not None and (
                not isinstance(self.hb_kvf_min, (int, float))
                or not (0.0 < float(self.hb_kvf_min) <= 10.0)):
            errors.append(
                f"hb_kvf_min (free-atmosphere diffusivity floor [m^2/s]) must "
                f"be None or finite in (0, 10]; got {self.hb_kvf_min!r}."
            )
        # ...and it is only consumed by a turbulence scheme whose nested
        # config carries a ``kvf_min`` field (holtslag_boville).  On any other
        # scheme _resolve_turbulence_config finds no field and the _replace
        # never happens, so the run would proceed with the knob silently
        # inert (codex review P2; same class as the k_h_scale lane guard).
        elif self.hb_kvf_min is not None and self.turbulence != "holtslag_boville":
            errors.append(
                f"hb_kvf_min is consumed only by the holtslag_boville "
                f"free-atmosphere diffusivity floor; with "
                f"turbulence={self.turbulence!r} the override reaches no "
                f"scheme field and would be silently inert."
            )
        # Slab-land interface flux law: membership + the unified law needs an
        # atmosphere-side surface-layer law to unify WITH.  turbulence="none"
        # has no turbulence-scheme surface layer, so "unified" would silently
        # degrade to the legacy slab bulk — reject loudly instead.  The
        # multilayer (Richards) land tile replaces the slab SEB entirely, so
        # the flag would be a silent no-op there — reject that too.
        _valid_land_iface = ("legacy_dual", "unified")
        if self.land_interface_flux not in _valid_land_iface:
            errors.append(
                f"land_interface_flux must be one of {_valid_land_iface}, "
                f"got {self.land_interface_flux!r}"
            )
        elif self.land_interface_flux == "unified":
            if self.turbulence == "none":
                errors.append(
                    "land_interface_flux='unified' requires an active "
                    "turbulence scheme (its surface layer IS the unified flux "
                    "law); with turbulence='none' the slab would silently "
                    "fall back to its own bulk law. Use the default "
                    "'legacy_dual' or enable a turbulence scheme."
                )
            if self.use_multilayer_land:
                errors.append(
                    "land_interface_flux='unified' applies to the SLAB land "
                    "SEB only; use_multilayer_land replaces the slab, so the "
                    "flag would be a silent no-op. Drop one of the two."
                )
            if not (self.land_mask_path
                    or (self.slab_land_active
                        and self.topography not in ("flat", "gaussian"))):
                errors.append(
                    "land_interface_flux='unified' needs an ACTIVE slab land "
                    "tile with actual land: pass land_mask_path, or "
                    "slab_land_active=True with a real topography "
                    f"(got topography={self.topography!r}, "
                    f"slab_land_active={self.slab_land_active!r}, no land mask). "
                    "Idealized topography derives f_land == 0 everywhere; "
                    "without an active land tile the slab SEB never steps "
                    "and the flag is a silent no-op."
                )
            # Mirror the model_driver.run() lane dispatch exactly: the MPAS
            # (grid_type-keyed) and spectral lanes run _run_mpas /
            # _run_spectral with combined.make_physics — the driver
            # PhysicsPipeline slab never steps there, so 'unified' would be
            # silently inert (codex R2: land_mask_path satisfied the tile
            # gate on MPAS without tripping the slab-flag rejection).
            if _is_mpas or d.discretization == "spectral":
                errors.append(
                    "land_interface_flux='unified' is consumed only by the "
                    "driver PhysicsPipeline slab (the compiled/per-step "
                    "lanes); the MPAS and spectral lanes build their physics "
                    "via combined.make_physics and never step the slab SEB — "
                    "the flag would be silently inert on "
                    f"discretization={d.discretization!r} / "
                    f"grid_type={g.grid_type!r}."
                )
            if self.physics_parameterization != "none":
                errors.append(
                    "land_interface_flux='unified' is not supported with "
                    "physics_parameterization='ml': the ML physics computes "
                    "its own surface fluxes (bypassing the turbulence "
                    "scheme's surface layer), so the slab would unify with a "
                    "law the atmosphere does not apply."
                )
            # The lat-lon operator-split SPMD lane builds a FRESH per-band
            # pipeline (model_driver.build_physics_pipeline on band_grid) and
            # never copies the post-construction land attributes that only
            # ever land on self.physics (f_land, slab_land_active).  The band
            # pipeline therefore sees land INACTIVE and skips the slab SEB
            # entirely, so the flag would be silently inert (codex review P1).
            # NB this is a property of the SPMD lane, not of this flag — slab
            # land does not step there at all; guarding the flag is in scope,
            # fixing that lane is not.
            if self.enable_latlon_spmd:
                errors.append(
                    "land_interface_flux='unified' is consumed by the slab "
                    "SEB, which the lat-lon operator-split SPMD lane never "
                    "steps: enable_latlon_spmd=True builds a fresh per-band "
                    "physics pipeline that never receives f_land / "
                    "slab_land_active, so the flag would be silently inert."
                )
        # Slab-land heat capacity [J/m^2/K]: must be positive-finite (the
        # semi-implicit denominator is C_land - dt*dflux_dT; C_land <= 0 flips
        # its sign and the update diverges).  Bounds span thin-skin (~1e4,
        # a few mm of soil) to full-column (~1e8, tens of m of water).
        if not (1.0e4 <= self.C_land <= 1.0e8):
            errors.append(
                f"C_land (slab-land heat capacity [J/m^2/K]) must be finite "
                f"in [1e4, 1e8]; got {self.C_land!r}."
            )
        # Soil-moisture init fraction of saturation: finite, in (0, 1].
        _soil_init_modes = ("aridity", "saturation_fraction")
        if self.land_soil_init not in _soil_init_modes:
            errors.append(
                f"land_soil_init must be one of {_soil_init_modes}, got "
                f"{self.land_soil_init!r}.")
        if not (0.0 < self.land_soil_moisture_init_frac <= 1.0):
            errors.append(
                f"land_soil_moisture_init_frac (theta_init/theta_sat) must be "
                f"finite and in (0, 1]; got {self.land_soil_moisture_init_frac!r}."
            )
        # Land surface-scheme membership (mirror the model_driver dispatch so a
        # typo fails here, not at run time).
        _valid_land_surface = ("simple_seb", "two_leaf", "clm_ml")
        if self.land_surface_scheme not in _valid_land_surface:
            errors.append(
                f"land_surface_scheme must be one of {_valid_land_surface}, "
                f"got {self.land_surface_scheme!r}"
            )
        # Stomatal soil-water limitation needs the bucket to supply beta_soil.
        if (self.land_stomatal_beta and not self.land_soil_bucket
                and not self.use_multilayer_land):
            errors.append(
                "land_stomatal_beta=True requires land_soil_bucket=True "
                "(the bucket supplies the soil availability beta_soil that the "
                "Jarvis stomatal model down-regulates) — UNLESS use_multilayer_land "
                "is set, in which case the Richards multilayer soil supplies the "
                "root-zone moisture availability instead (the stomata are threaded "
                "into MultiLayerLandConfig.stomata, PR #715)."
            )
        # Morrison ice-process scalar bounds (mirror MorrisonConfig's
        # __param_spec__ so an out-of-range or non-finite knob fails at
        # config time, not deep in a run — codex 2026-07-26 morrison-wiring
        # review: validate_strict accepted NaN and out-of-spec values).
        _morrison_touched = []
        for _f, _lo, _hi in (
            ("morrison_bergeron_rate", 1.0e-4, 1.0e-2),
            ("morrison_rime_coeff", 0.0, 2.0),
            ("morrison_dep_coeff", 1.0e-4, 1.0e-2),
            ("morrison_agg_coeff", 1.0e-4, 1.0e-2),
            ("morrison_k_au", 50.0, 5000.0),
            ("morrison_fall_a_i", 230.0, 6300.0),
            ("morrison_ice_snow_d_auto", 8.0e-5, 8.0e-4),
            ("morrison_hom_ice_nuc_N", 1.0e4, 1.0e7),
        ):
            _v = getattr(self, _f)
            if not math.isfinite(_v) or not (_lo <= _v <= _hi):
                raise ValueError(
                    f"{_f}={_v} outside MorrisonConfig spec bounds "
                    f"[{_lo}, {_hi}] (or non-finite)")
            # SAME tolerance as the resolver threading (rel_tol=1e-6):
            # a float32-noise value must be "default" in BOTH places, never
            # rejected here while the resolver would apply nothing (codex
            # 2026-07-26 round 3, item 4).
            if not math.isclose(_v, ExperimentConfig._field_defaults[_f],
                                rel_tol=1e-6, abs_tol=0.0):
                _morrison_touched.append(_f)
        # A touched morrison_* scalar on any OTHER scheme must be refused at
        # CONFIG time: microphysics='none' early-returns before either lane's
        # overlay, so the resolver-level hard gate never sees it and the knob
        # would be silently inert (codex 2026-07-26 round 2, item 3 — the
        # exact silent-drop class this wiring exists to close).
        if _morrison_touched and self.microphysics != "morrison":
            raise ValueError(
                f"morrison_* overrides {_morrison_touched} require "
                f"microphysics='morrison' (got {self.microphysics!r}); on "
                "any other scheme they would be silently inert or retune a "
                "foreign scheme. Drop them or switch schemes.")
        # morrison_hom_ice_nuc_N is read ONLY inside the homogeneous-
        # nucleation branch: overriding it with the flag off would be the
        # exact silently-inert class the scheme gate above closes.
        if ("morrison_hom_ice_nuc_N" in _morrison_touched
                and not self.homogeneous_ice_nucleation):
            raise ValueError(
                "morrison_hom_ice_nuc_N override requires "
                "homogeneous_ice_nucleation=True — the leaf is consumed only "
                "by the hom-nucleation branch and would be silently inert.")
        if self.morrison_flavor not in ("mg", "sam"):
            raise ValueError(
                f"morrison_flavor={self.morrison_flavor!r} unknown; choose "
                "'mg' (E3SM MG, default) or 'sam' (gSAM M2005).")
        _nmm_max = self.morrison_sed_cfl_substeps_max
        if (not isinstance(_nmm_max, int) or isinstance(_nmm_max, bool)
                or not 1 <= _nmm_max <= _sed_substeps_cap_limit()):
            errors.append(
                "morrison_sed_cfl_substeps_max must be an int in "
                f"[1, {_sed_substeps_cap_limit()}] (the sub-step loop costs "
                f"linearly in it), got {_nmm_max!r}")
        if (_nmm_max != ExperimentConfig._field_defaults[
                "morrison_sed_cfl_substeps_max"] and self.microphysics != "morrison"):
            errors.append(
                f"morrison_sed_cfl_substeps_max={_nmm_max} requires "
                f"microphysics='morrison' (got {self.microphysics!r})")
        for _nm in ("morrison_sed_cfl_substeps", "morrison_sed_cfl_substeps_strict"):
            _v = getattr(self, _nm)
            if not isinstance(_v, bool):
                errors.append(f"{_nm} must be a bool, got {_v!r}")
            elif (_v is not ExperimentConfig._field_defaults[_nm]
                    and self.microphysics != "morrison"):
                errors.append(
                    f"{_nm}={_v} requires microphysics='morrison' "
                    f"(got {self.microphysics!r}); it would be silently inert")
        if (self.morrison_sed_cfl_substeps_strict is True
                and self.morrison_sed_cfl_substeps is False):
            errors.append(
                "morrison_sed_cfl_substeps_strict=True needs "
                "morrison_sed_cfl_substeps=True (nothing to check otherwise)")
        if self.morrison_flavor != "mg" and self.microphysics != "morrison":
            raise ValueError(
                f"morrison_flavor={self.morrison_flavor!r} requires "
                f"microphysics='morrison' (got {self.microphysics!r}); on any "
                "other scheme the flavor would be silently inert.")
        # Optional cloud-tuning override bounds (track CloudConfig.__param_spec__
        # so an out-of-range knob fails early, not deep in the cloud diagnosis).
        # NOT a byte-for-byte mirror: __param_spec__ additionally seeds the
        # TRAINING sigmoid re-parameterisation (param_collector._seed_raw), so a
        # bound only the forward driver needs is widened HERE alone.
        for _f, _lo, _hi in (
            ("cloud_rh_crit", 0.5, 0.99),
            # Lower bound 5.0e-5 -> 1.0e-6.  The radiative condensate floor
            # enters additively as ``cf_strat * q_liq_incloud`` /
            # ``cf_strat * q_c_diagnostic`` (cloud_fraction.py ``q_floor_liq`` /
            # ``q_floor_ice``; under diagnostic_condensate_scheme='adiabatic' the
            # LIQUID part is depth-derived and CAPPED at q_c_diagnostic instead).
            # It is never itself a denominator, and the downstream quotients that
            # see the resulting condensate are floor-protected far below 1e-6
            # (PSD radius ``q_c_pos`` at 1e-15, cloud_fraction.py:872; the
            # adiabatic liquid split at 1e-30, cloud_fraction.py:824).  So any
            # strictly positive value is numerically safe, and >0 keeps this a
            # floor rather than "off".
            # The old 5.0e-5 was the *floor of the tuning range*, not a physical
            # limit.  The MPAS AMIP campaign ran exactly ON it while carrying a
            # ~+32 W/m2 reflected-shortwave excess that an offline
            # production-fidelity RRTMGP factorial attributes almost entirely to
            # this knob (5e-5 -> 0 is -35.6 W/m2, ~103% of the gap; the ladder
            # rungs 2e-5/1e-5/5e-6 give -8.8/-15.6/-22.4).  Testing those rungs
            # coupled requires the bound to admit them.  Evidence is in the run
            # directory, not the repo: report ``floor_attribution_hifi.md``.
            # NOT widened alongside it: CloudConfig.__param_spec__ keeps
            # (5.0e-5, 1.5e-3), because those bounds ALSO seed the training
            # sigmoid re-parameterisation (param_collector._seed_raw), where a
            # widening is NOT inert.  Sub-5e-5 therefore reaches the model via
            # ``--q-c-diagnostic`` or a ``--config`` YAML, but NOT via ``--params``.
            ("cloud_q_c_diagnostic", 1.0e-6, 1.0e-3),
            ("cloud_conv_cloud_max", 0.1, 1.0),
            ("cloud_conv_cloud_condensate", 1.0e-5, 1.0e-3),
            ("cloud_inhomogeneity_factor", 0.3, 1.0),
            ("cloud_fsd", 0.0, 1.0),
            ("cloud_p_xr", 0.05, 1.0),
            ("cloud_alpha_xr", 10.0, 1000.0),
            ("cloud_cover_condensate_q_ref", 1.0e-6, 1.0e-3),
            ("cloud_cap_floor_lat_deg", 40.0, 89.0),
            ("cloud_cap_floor_p_max_pa", 20000.0, 100000.0),
            ("cloud_cap_floor_cf", 0.1, 1.0),
            ("cloud_cap_floor_q_c", 1.0e-6, 1.0e-3),
            ("snow_age_activation_K", 0.0, 20000.0),
            # 0.5 d = melting spring snow; 400 d spans the cold-plateau
            # timescale the literature supports (Warren & Wiscombe 1980).
            ("land_snow_tau_days", 0.5, 400.0),
            ("cloud_adiabatic_lwc_rate", 5.0e-7, 3.0e-6),
            # 0 = the scheme's own "no limit"; 1e3 Pa is above any tropopause,
            # 5e4 Pa would switch the boundary-layer scheme off in the free
            # troposphere, which is not what a taper is for.
            ("clubb_trop_cloud_top_press", 0.0, 5.0e4),
            # 1 = byte-identical; the probe's own spec bounds (CLUBBConfig).
            ("clubb_q_flux_scale", 0.1, 10.0),
        ):
            _v = getattr(self, _f)
            # 0.0 means OFF for the two radiative condensate floors.  Their
            # declared ranges start above zero because a floor of 0 was
            # previously unreachable, so without this the liquid-partition
            # guard would demand a value this loop then rejects (codex).
            #
            # UNCONDITIONAL, deliberately.  Gating it on the partition made
            # "floors off, partition off" unbuildable, which is exactly the
            # control arm needed to attribute anything to the partition: the
            # two would have had to move together and no measurement could
            # separate them.  A guard that forbids the control is a defect in
            # the guard.  This only ADDS a previously-refused configuration;
            # every existing deck keeps its value and its behaviour.
            if (_v == 0.0 and _f in ("cloud_q_c_diagnostic",
                                     "cloud_conv_cloud_condensate")):
                continue
            if _v is not None and not (_lo <= _v <= _hi):
                errors.append(
                    f"{_f}={_v!r} out of range [{_lo}, {_hi}]"
                )
        if self.turbulence_override is not None:
            from legoesm.atmosphere.physics.turbulence.config import (
                TurbulenceConfig,
            )
            if not isinstance(self.turbulence_override, TurbulenceConfig):
                errors.append(
                    "turbulence_override must be a TurbulenceConfig, got "
                    f"{type(self.turbulence_override).__name__}"
                )
            elif self.turbulence_override.scheme != self.turbulence:
                errors.append(
                    f"turbulence_override.scheme={self.turbulence_override.scheme!r} "
                    f"must equal turbulence={self.turbulence!r} (an override refines "
                    f"the same scheme's sub-config, it does not switch schemes)"
                )
        if self.gravity_wave_drag_override is not None:
            from legoesm.atmosphere.physics.gravity_wave_drag.config import (
                GravityWaveDragConfig,
            )
            if not isinstance(
                self.gravity_wave_drag_override, GravityWaveDragConfig
            ):
                errors.append(
                    "gravity_wave_drag_override must be a GravityWaveDragConfig, "
                    f"got {type(self.gravity_wave_drag_override).__name__}"
                )
            elif (self.gravity_wave_drag_override.scheme
                  != self.gravity_wave_drag):
                errors.append(
                    "gravity_wave_drag_override.scheme="
                    f"{self.gravity_wave_drag_override.scheme!r} must equal "
                    f"gravity_wave_drag={self.gravity_wave_drag!r} (an override "
                    "refines the same scheme's sub-config, it does not switch "
                    "schemes)"
                )
        # GWD scalars that ``gwd_config_for`` overlays onto the kernel leaves.
        # Every one is a strictly-positive physical quantity (a wavenumber, a
        # spreading factor, a stress/flux cap, an rms launch wind), and none is
        # bounds-checked anywhere else on the CLI/--config route (the
        # ``__param_spec__`` bounds only gate the ``--params`` loader).  Silent
        # failure modes without this guard: ``hines_Fmax < 0`` makes
        # ``jnp.clip(drag, 0.0, Fmax)`` return the NEGATIVE cap at every level
        # (constant spurious drag, no error), and ``hines_total_rms_wind <= 0``
        # zeroes the amplitude growth so the scheme silently does nothing.
        # Positivity + finiteness only — the calibratable RANGE stays in
        # ``__param_spec__`` so it is not maintained twice.
        for _f in ("mcfarlane_k_wave", "mcfarlane_directional_spread",
                   "mcfarlane_tau_max", "hines_total_rms_wind", "hines_Fmax",
                   "e3sm_cam_taubgnd", "e3sm_cam_c0", "e3sm_cam_launch_p",
                   "e3sm_cam_effgw"):
            _v = getattr(self, _f)
            if not math.isfinite(_v) or _v <= 0.0:
                errors.append(
                    f"{_f} must be a positive, finite gravity-wave-drag "
                    f"parameter, got {_v!r}"
                )
        _e3sm_parts = str(self.e3sm_cam_source).split("+")
        _e3sm_valid = ("orographic", "frontal", "convective", "background")
        if self.e3sm_cam_source not in _e3sm_valid and (
                any(p not in _e3sm_valid for p in _e3sm_parts)
                or len(set(_e3sm_parts)) != len(_e3sm_parts)):
            errors.append(
                f"e3sm_cam_source must be one of {_e3sm_valid} or a "
                f"'+'-joined set of distinct ones, got {self.e3sm_cam_source!r}"
            )
        if ("convective" in _e3sm_parts
                and "e3sm_cam" in str(self.gravity_wave_drag).split("+")
                and self.convection == "none"):
            errors.append(
                "e3sm_cam_source includes 'convective' (Beres) but "
                "convection='none': no scheme publishes the convective "
                "heating the Beres source reads, so it would launch nothing "
                "(silent no-op)"
            )
        _fg = self.e3sm_cam_frontgfc
        if _fg is not None and (not math.isfinite(_fg) or _fg <= 0.0):
            errors.append(
                f"e3sm_cam_frontgfc must be None or a positive, finite "
                f"frontogenesis threshold [K^2/(m^2 s)], got {_fg!r}")
        for _f in ("e3sm_cam_effgw_cm", "e3sm_cam_effgw_beres"):
            _v = getattr(self, _f)
            if _v is not None and (
                    not math.isfinite(_v) or _v <= 0.0 or _v > 1.0):
                errors.append(
                    f"{_f} must be None or a finite efficiency in (0, 1], "
                    f"got {_v!r}"
                )
        if not isinstance(self.e3sm_cam_dttke_intrinsic, bool):
            errors.append(
                f"e3sm_cam_dttke_intrinsic must be a bool, "
                f"got {self.e3sm_cam_dttke_intrinsic!r}"
            )
        if self.e3sm_cam_beres_variant not in ("e3sm", "cam6"):
            errors.append(
                f"e3sm_cam_beres_variant must be one of ('e3sm', 'cam6'), "
                f"got {self.e3sm_cam_beres_variant!r}"
            )
        if self.e3sm_cam_mfcc_table_path:
            if not os.path.isfile(self.e3sm_cam_mfcc_table_path):
                errors.append(
                    f"e3sm_cam_mfcc_table_path={self.e3sm_cam_mfcc_table_path!r} "
                    f"does not exist (the Beres table would silently fall "
                    f"back to nothing at build time)"
                )
        if isinstance(self.e3sm_cam_pgwv, bool) or \
                not isinstance(self.e3sm_cam_pgwv, int) or \
                self.e3sm_cam_pgwv < 0:
            errors.append(
                f"e3sm_cam_pgwv must be an int >= 0, "
                f"got {self.e3sm_cam_pgwv!r}"
            )
        if any(p in ("frontal", "background") for p in _e3sm_parts) and \
                self.e3sm_cam_pgwv < 1:
            errors.append(
                f"e3sm_cam_pgwv must be >= 1 for a launch-everywhere "
                f"spectrum; pgwv 0 is the single c=0 wave, got "
                f"{self.e3sm_cam_pgwv!r}"
            )
        if not math.isfinite(self.e3sm_cam_effgw) or \
                self.e3sm_cam_effgw > 1.0:
            errors.append(
                f"e3sm_cam_effgw must be a finite efficiency <= 1.0, "
                f"got {self.e3sm_cam_effgw!r}"
            )
        if not math.isfinite(self.e3sm_cam_taubgnd) or \
                self.e3sm_cam_taubgnd > 1.0e-2:
            errors.append(
                f"e3sm_cam_taubgnd must be a finite base stress <= 1.0e-2 Pa, "
                f"got {self.e3sm_cam_taubgnd!r}"
            )
        if not isinstance(self.e3sm_cam_latitude_taper, bool):
            errors.append(
                f"e3sm_cam_latitude_taper must be a bool, "
                f"got {self.e3sm_cam_latitude_taper!r}"
            )
        _valid_gwd = VALID_GWD
        # A ``+``-joined string composes multiple GWD sources whose tendencies
        # are summed — orographic (mcfarlane/lindzen) and non-orographic
        # (hines/rayleigh/prognostic_spectral) parameterize distinct wave
        # populations and are run together in CMIP-class GCMs
        # (e.g. ``hines+mcfarlane`` or ``mcfarlane+prognostic_spectral``,
        # issue #834).  ``prognostic_spectral`` is the one STATEFUL composable
        # source — its wave-action spectrum threads through the physics carry,
        # so at most one stateful source may appear.  ``e3sm_cam`` /
        # ``ml_emulator`` need extra per-column source fields / a network
        # module the composite path does not carry and are NOT composable.
        _composable_stateless = ("rayleigh", "lindzen", "mcfarlane", "hines",
                                 "e3sm_cam")
        _composable_stateful = ("prognostic_spectral",)
        _composable = _composable_stateless + _composable_stateful
        _gwd_parts = self.gravity_wave_drag.split("+")
        if len(_gwd_parts) > 1:
            bad = [p for p in _gwd_parts if p not in _composable]
            if bad:
                errors.append(
                    f"composite gravity_wave_drag parts must each be one of "
                    f"{_composable}, got invalid {bad} in "
                    f"{self.gravity_wave_drag!r}"
                )
            # The EFFECTIVE source is the override's when one is supplied:
            # gwd_config_for returns gravity_wave_drag_override untouched, so
            # gating on the scalar would reject a valid override (codex #5).
            _ov = getattr(self, "gravity_wave_drag_override", None)
            _e3sm_src = (getattr(getattr(_ov, "e3sm_cam", None), "source", None)
                         if _ov is not None else None) or self.e3sm_cam_source
            if ("e3sm_cam" in _gwd_parts
                    and "orographic" in str(_e3sm_src).split("+")
                    and any(p in ("lindzen", "mcfarlane") for p in _gwd_parts)):
                errors.append(
                    f"composite gravity_wave_drag {self.gravity_wave_drag!r} "
                    f"pairs e3sm_cam's orographic source with lindzen/mcfarlane "
                    f"(e3sm_cam_source={_e3sm_src!r}): topographic drag would "
                    f"be double-counted; use frontal/convective/background "
                    f"sources there, or run e3sm_cam alone with "
                    f"'orographic+frontal+convective'"
                )
            # Mirror get_gwd_fn's runtime rule so a duplicate composite
            # fails HERE, not later during physics construction.
            if len(set(_gwd_parts)) != len(_gwd_parts):
                errors.append(
                    f"composite gravity_wave_drag has duplicate parts: "
                    f"{self.gravity_wave_drag!r}"
                )
            _n_stateful = sum(p in _composable_stateful for p in _gwd_parts)
            if _n_stateful > 1:
                errors.append(
                    f"composite gravity_wave_drag may contain at most one "
                    f"stateful source {_composable_stateful} (its wave-action "
                    f"spectrum is a single carry), got {_n_stateful} in "
                    f"{self.gravity_wave_drag!r}"
                )
        elif self.gravity_wave_drag not in _valid_gwd:
            errors.append(
                f"gravity_wave_drag must be one of {_valid_gwd} "
                f"(or a '+'-joined composite of {_composable}), "
                f"got {self.gravity_wave_drag!r}"
            )
        # Checkpoint serialization format (mirror the io/restart writer set so a
        # typo fails here instead of silently writing the wrong format).
        _valid_checkpoint_format = ("npz", "zarr")
        if self.output.checkpoint_format not in _valid_checkpoint_format:
            errors.append(
                f"checkpoint_format must be one of {_valid_checkpoint_format}, "
                f"got {self.output.checkpoint_format!r}"
            )
        # Reject unsupported coupled/ESM modes with actionable errors.
        if self.carbon_cycle != "none":
            errors.append(
                f"carbon_cycle={self.carbon_cycle!r} is not implemented. "
                f"ModelDriver is atmosphere-only with prescribed SST/SIC. "
                f"Set carbon_cycle='none' or use a coupled driver."
            )
        _valid_ic = ("default", "standard", "era5")
        if self.ic not in _valid_ic:
            errors.append(f"ic must be one of {_valid_ic}, got {self.ic!r}")
        if self.ic == "era5" and not self.ic_path:
            errors.append("ic='era5' requires ic_path to be set")
        if self.ic == "standard":
            # The standard-atmosphere IC overrides a grid-space temperature
            # Field AND (lat-lon only) a geographic (eastward) thermal-wind jet.
            #   * lat-lon: the A-grid u IS geographic-east, so the realistic T/
            #     p_s AND the balanced jet are assigned directly.
            #   * cubed_sphere: the realistic T / p_s overlay is grid-agnostic
            #     and IS applied; the balanced jet is SKIPPED (u/v are cube-LOCAL
            #     components that would need a per-cell grid-angle rotation) — a
            #     coupled climate spin-up grows its own circulation from the T
            #     gradient (no balanced-jet IC needed, unlike a baroclinic-wave
            #     test).  See ModelDriver._apply_standard_atmosphere_ic.
            # Still not wired (fail early, before setup):
            #   * gaussian/spectral: temperature lives in spectral space (T_hat),
            #     no grid-space T Field;
            #   * mpas: not wired.
            _gt_std = normalize_grid_type(self.grid.grid_type)
            if _gt_std not in ("latlon", "cubed_sphere"):
                errors.append(
                    f"ic='standard' is implemented for grid_type "
                    f"'latlon' and 'cubed_sphere'; got "
                    f"grid_type={self.grid.grid_type!r} "
                    f"(discretization={self.dycore.discretization!r}). "
                    f"Use ic='default', or ic='era5' for spectral/mpas."
                )
            # T_init is the equator surface temperature; the pole is
            # T_init - 40 K (StandardAtmosphereConfig.equator_pole_delta_K). A
            # too-cold T_init drives the pole surface temperature non-positive
            # and would NaN the thermal-wind setup, so require a physical
            # equator surface temperature here (fail-early, before setup).
            if not (150.0 <= self.T_init <= 360.0):
                errors.append(
                    f"ic='standard' requires a physical equator surface "
                    f"temperature 150 K <= T_init <= 360 K; got "
                    f"T_init={self.T_init} K."
                )

        if self.aimip_variant not in AIMIP_VARIANTS:
            errors.append(
                f"aimip_variant must be one of {AIMIP_VARIANTS}, "
                f"got {self.aimip_variant!r}"
            )
        # NOTE: the classical variant runs through the AIMIP training entry
        # point (make_aimip_classical_spectral_physics), NOT the unified
        # ModelDriver pipeline — tiedtke is profile-prognostic and
        # build_physics_pipeline fails fast on it with a pointer at the
        # bridge factory (see _PIPELINE_UNSUPPORTED_CONVECTION).
        if self.aimip_variant == "classical":
            if self.convection != "tiedtke":
                errors.append(
                    "aimip_variant='classical' requires convection='tiedtke', "
                    f"got {self.convection!r}"
                )
            if self.turbulence != "louis":
                errors.append(
                    "aimip_variant='classical' requires turbulence='louis', "
                    f"got {self.turbulence!r}"
                )
            if self.gravity_wave_drag != "mcfarlane":
                errors.append(
                    "aimip_variant='classical' requires "
                    "gravity_wave_drag='mcfarlane', "
                    f"got {self.gravity_wave_drag!r}"
                )
            if self.cloud_scheme != "xu_randall":
                errors.append(
                    "aimip_variant='classical' requires "
                    "cloud_scheme='xu_randall', "
                    f"got {self.cloud_scheme!r}"
                )

        if self.output.evaluation.enabled:
            ceval_py = self.output.evaluation.climateeval_python
            if not ceval_py or not (
                Path(ceval_py).is_file() and os.access(ceval_py, os.X_OK)
            ):
                errors.append(
                    "output.evaluation.enabled=True requires "
                    "climateeval_python to point at a real, executable "
                    f"ClimateEval Python interpreter (got {ceval_py!r}). "
                    "ClimateEval is a separate, externally-installed tool "
                    "(never a legoESM dependency) — set "
                    "--evaluation-climateeval-python or the "
                    "LEGOESM_CLIMATEEVAL_PYTHON environment variable. See "
                    "docs/user-guide/climateeval_evaluation.md for setup."
                )
            data_root = self.output.evaluation.data_root_dir
            if not data_root or not Path(data_root).is_dir():
                errors.append(
                    "output.evaluation.enabled=True requires data_root_dir "
                    f"to point at a real ClimateEval reference-data "
                    f"directory (got {data_root!r}). Set "
                    "--evaluation-data-root-dir or the "
                    "LEGOESM_CLIMATEEVAL_DATA_ROOT environment variable. "
                    "See docs/user-guide/climateeval_evaluation.md."
                )

        # Seasonal insolation alignment (radiation-only; see the field doc).
        if self.insolation_start_doy is not None:
            _doy = self.insolation_start_doy
            if not (isinstance(_doy, (int, float)) and 1.0 <= float(_doy) < 366.0):
                errors.append(
                    "insolation_start_doy must be None or a noleap day-of-year "
                    f"in [1, 366), got {self.insolation_start_doy!r}"
                )

        if errors:
            raise ValueError(
                "Invalid ExperimentConfig:\n  " + "\n  ".join(errors)
            )

    def validate(self) -> list[str]:
        """Check for suspicious parameter combinations.

        Returns a list of warning strings (empty if config is clean).
        """
        warns: list[str] = []
        if self.radiation == "gray" and self.aerosol_forcing != "off":
            warns.append("aerosol_forcing is ignored with gray radiation")
        if self.microphysics != "none" and self.radiation == "gray":
            warns.append(
                "microphysics without RRTMGP radiation may give unrealistic results"
            )
        if self.dycore.dt > 900 and self.grid.resolution >= 32:
            warns.append(
                f"dt={self.dycore.dt}s may violate CFL at C{self.grid.resolution}"
            )
        if self.cloud_scheme != "none" and self.radiation == "gray":
            warns.append(
                "cloud_scheme is ignored with gray radiation; "
                "set radiation='rrtmgp' for cloud-radiation coupling"
            )
        if self.fix_moisture and self.microphysics != "none":
            # The current ``fix_moisture_hydrostatic`` implementation only
            # rescales ``q_v``, not prognostic condensate (``q_c``/``q_r``)
            # nor cumulative precipitation flux at the surface.  When a
            # precipitating microphysics scheme is active, the fixer
            # multiplies q_v back up after each precipitation event — an
            # unphysical source of water vapor that compounds with the
            # microphysical condensation/heating loop and drives the
            # column unstable (catalogued under AMIP.md "Known issues").
            warns.append(
                "fix_moisture with prognostic-condensate microphysics "
                f"({self.microphysics}) is INCORRECT: the current "
                "implementation rescales only q_v, not q_c/q_r/precip — "
                "spurious vapor sources will accumulate and may drive the "
                "column unstable.  Disable --fix-moisture or replace it "
                "with a fix_total_water path that tracks precipitation."
            )
        _valid_perf_modes = ("auto", "always", "never")
        if self.output.diagnostics_perf_mode not in _valid_perf_modes:
            raise ValueError(
                f"diagnostics_perf_mode must be one of {_valid_perf_modes}, "
                f"got {self.output.diagnostics_perf_mode!r}"
            )
        if (self.output.cmip_output
                and self.output.diagnostics_perf_mode == "always"):
            warns.append(
                "diagnostics_perf_mode='always' is incompatible with "
                "cmip_output=True; perf mode will be disabled at runtime "
                "to ensure CMIP accumulation is not skipped"
            )
        if self.physics_parameterization == "ml":
            if self.convection != "mass_flux" or self.turbulence != "louis":
                warns.append(
                    "physics_parameterization='ml' currently expects "
                    "convection='mass_flux' and turbulence='louis'"
                )
            if self.microphysics not in ("none", "kessler", "sundqvist"):
                warns.append(
                    "physics_parameterization='ml' currently supports "
                    "microphysics='none', 'kessler', or 'sundqvist'"
                )
            has_ckpt = bool(self.physics_parameterization_checkpoint)
            has_stats = bool(self.physics_parameterization_stats)
            if has_ckpt != has_stats:
                warns.append(
                    "physics_parameterization='ml' expects both "
                    "physics_parameterization_checkpoint and "
                    "physics_parameterization_stats"
                )
        return warns

    # ------------------------------------------------------------------
    # Legacy AMIP adapter (serialization boundary only)
    # ------------------------------------------------------------------

    @staticmethod
    def from_amip_config(amip_cfg) -> ExperimentConfig:
        """Convert AMIPExperimentConfig to ExperimentConfig.

        Used at deserialization boundaries (checkpoint load, legacy
        experiment factory) — not in core runtime paths.
        """
        grid = GridConfig(
            resolution=amip_cfg.resolution,
            nlev=amip_cfg.nlev,
            vertical_coord=amip_cfg.vertical_coord,
            p_top_Pa=amip_cfg.p_top_Pa,
            stretching=amip_cfg.stretching,
            tropopause_refine=getattr(amip_cfg, 'tropopause_refine', 1.0),
        )
        dycore = DycoreConfig(
            dt=amip_cfg.dt,
            hyperdiff_scale=getattr(amip_cfg, 'hyperdiff_scale', 1.0),
            a_h_scale=getattr(amip_cfg, 'a_h_scale', 1.0),
        )
        output = OutputConfig(
            output_dir=getattr(amip_cfg, 'output_dir', ''),
            diag_days=amip_cfg.diag_days,
            checkpoint_days=amip_cfg.checkpoint_days,
            monthly_means=getattr(amip_cfg, 'monthly_means', False),
            cmip_output=getattr(amip_cfg, 'cmip_output', False),
            clear_sky_diag=getattr(amip_cfg, 'clear_sky_diag', False),
            budget_ledger=getattr(amip_cfg, 'budget_ledger', False),
            budget_ledger_sigma_band=getattr(
                amip_cfg, 'budget_ledger_sigma_band', None),
        )
        return ExperimentConfig(
            grid=grid,
            dycore=dycore,
            output=output,
            days=amip_cfg.days,
            start_day=amip_cfg.start_day,
            dataset=amip_cfg.dataset,
            forcing_path=amip_cfg.forcing_path,
            sic_path=getattr(amip_cfg, 'sic_path', ''),
            sst_var=amip_cfg.sst_var,
            sic_var=amip_cfg.sic_var,
            time_var=getattr(amip_cfg, 'time_var', ''),
            lat_var=getattr(amip_cfg, 'lat_var', ''),
            lon_var=getattr(amip_cfg, 'lon_var', ''),
            sst_offset=amip_cfg.sst_offset,
            sic_scale=amip_cfg.sic_scale,
            radiation=amip_cfg.radiation,
            rad_update_steps=amip_cfg.rad_update_steps,
            physics_update_steps=amip_cfg.physics_update_steps,
            cld_macmic_num_steps=amip_cfg.cld_macmic_num_steps,
            morrison_sed_cfl_substeps=amip_cfg.morrison_sed_cfl_substeps,
            morrison_sed_cfl_substeps_max=amip_cfg.morrison_sed_cfl_substeps_max,
            morrison_sed_cfl_substeps_strict=amip_cfg.morrison_sed_cfl_substeps_strict,
            unfused_radiation=getattr(amip_cfg, 'unfused_radiation', False),
            diurnal_cycle=amip_cfg.diurnal_cycle,
            co2_ppmv=amip_cfg.co2_ppmv,
            ch4_ppbv=amip_cfg.ch4_ppbv,
            n2o_ppbv=amip_cfg.n2o_ppbv,
            S_0=amip_cfg.S_0,
            ozone_source=amip_cfg.ozone_source,
            ozone_forcing=getattr(amip_cfg, 'ozone_forcing', 'inline'),
            ozone_file=getattr(amip_cfg, 'ozone_file', ''),
            ghg_forcing=getattr(amip_cfg, 'ghg_forcing', 'constant'),
            ghg_file=getattr(amip_cfg, 'ghg_file', ''),
            solar_source=getattr(amip_cfg, 'solar_source', 'constant'),
            solar_file=getattr(amip_cfg, 'solar_file', ''),
            solar_tsi_var=getattr(amip_cfg, 'solar_tsi_var', 'tsi'),
            solar_spectral_var=getattr(amip_cfg, 'solar_spectral_var', 'solar_fraction_by_gpt'),
            solar_spectral_band_order=getattr(amip_cfg, 'solar_spectral_band_order', 'auto'),
            aerosol_forcing=getattr(amip_cfg, 'aerosol_forcing', 'off'),
            aerosol_file=getattr(amip_cfg, 'aerosol_file', ''),
            aerosol_reference_aod=getattr(amip_cfg, 'aerosol_reference_aod', 0.03),
            volcanic_aerosol_file=getattr(amip_cfg, 'volcanic_aerosol_file', ''),
            volcanic_aerosol_scale=getattr(amip_cfg, 'volcanic_aerosol_scale', 1.0),
            volcanic_aerosol_lw=getattr(amip_cfg, 'volcanic_aerosol_lw', False),
            cloud_scheme=amip_cfg.cloud_scheme,
            microphysics=amip_cfg.microphysics,
            convection=getattr(amip_cfg, 'convection', 'sbm'),
            turbulence=getattr(amip_cfg, 'turbulence', 'none'),
            gravity_wave_drag=getattr(amip_cfg, 'gravity_wave_drag', 'none'),
            fix_moisture=getattr(amip_cfg, 'fix_moisture', False),
            energy_consistent_moisture_clip=getattr(
                amip_cfg, 'energy_consistent_moisture_clip', False),
            moisture_advection=getattr(amip_cfg, 'moisture_advection', False),
            topography=amip_cfg.topography,
            topo_smoothing=amip_cfg.topo_smoothing,
            topo_edge_blend=amip_cfg.topo_edge_blend,
            land_mask_path=getattr(amip_cfg, 'land_mask_path', ''),
            albedo_land_path=getattr(amip_cfg, 'albedo_land_path', ''),
            albedo_land_month=getattr(amip_cfg, 'albedo_land_month', 0),
            C_land=getattr(amip_cfg, 'C_land', 2.0e5),
            # Legacy checkpoints predate the flag and ran the legacy dual-law
            # slab, so the default is the correct restart behaviour.
            land_interface_flux=getattr(
                amip_cfg, 'land_interface_flux', 'legacy_dual'),
            emissivity_land=getattr(amip_cfg, 'emissivity_land', constants.emissivity_land),
            beta_land=getattr(amip_cfg, 'beta_land', 1.0),
            use_multilayer_land=getattr(amip_cfg, 'use_multilayer_land', False),
            multilayer_n_layers=getattr(amip_cfg, 'multilayer_n_layers', 10),
            multilayer_soil_depth=getattr(amip_cfg, 'multilayer_soil_depth', 3.0),
            era5_land_ic_path=getattr(amip_cfg, 'era5_land_ic_path', ''),
            cloud_r_eff_ice=getattr(amip_cfg, 'cloud_r_eff_ice', 30.0e-6),
            cloud_r_eff_liq_ocean=getattr(amip_cfg, 'cloud_r_eff_liq_ocean', 10.0e-6),
            cloud_r_eff_liq_land=getattr(amip_cfg, 'cloud_r_eff_liq_land', 7.0e-6),
            surfdata_path=getattr(amip_cfg, 'surfdata_path', ''),
            T_init=amip_cfg.T_init,
            rh_init=amip_cfg.rh_init,
            dynamic_albedo=amip_cfg.dynamic_albedo,
            carbon_cycle=amip_cfg.carbon_cycle,
            experiment=amip_cfg.experiment,
            start_year=amip_cfg.start_year,
            C_H=amip_cfg.C_H,
            C_E=amip_cfg.C_E,
            T_ice=amip_cfg.T_ice,
            albedo_ice=amip_cfg.albedo_ice,
            albedo_ocean=amip_cfg.albedo_ocean,
            sfc_emissivity=amip_cfg.sfc_emissivity,
            emissivity_ice=amip_cfg.emissivity_ice,
            tau_equator=amip_cfg.tau_equator,
            tau_pole=amip_cfg.tau_pole,
            tau_moist_coeff=getattr(amip_cfg, 'tau_moist_coeff', 0.0115),
            linear_frac=getattr(amip_cfg, 'linear_frac', 0.2),
            lw_diff_factor=getattr(amip_cfg, 'lw_diff_factor', 1.66),
            sw_tau_0=getattr(amip_cfg, 'sw_tau_0', 0.22),
            sw_exponent=getattr(amip_cfg, 'sw_exponent', 2.0),
            sundqvist_auto_rate=getattr(amip_cfg, 'sundqvist_auto_rate', 1e-3),
            sundqvist_evap_coeff=getattr(amip_cfg, 'sundqvist_evap_coeff', 5e-4),
            cloud_rh_crit=getattr(amip_cfg, 'cloud_rh_crit', 0.7),
            cloud_r_eff_liq=getattr(amip_cfg, 'cloud_r_eff_liq', 10.0e-6),
            sundqvist_sigmoid_sharpness=getattr(amip_cfg, 'sundqvist_sigmoid_sharpness', 20.0),
            sbm_T_min_convect=getattr(amip_cfg, 'sbm_T_min_convect', 200.0),
            louis_l_mix_max=getattr(amip_cfg, 'louis_l_mix_max', 100.0),
            louis_Ck=getattr(amip_cfg, 'louis_Ck', 0.4),
            louis_Ri_crit=getattr(amip_cfg, 'louis_Ri_crit', 0.25),
            louis_b_louis=getattr(amip_cfg, 'louis_b_louis', 5.0),
            louis_c_louis=getattr(amip_cfg, 'louis_c_louis', 16.6),
            louis_d_louis=getattr(amip_cfg, 'louis_d_louis', 5.0),
            louis_z0=getattr(amip_cfg, 'louis_z0', 1.0e-4),
            louis_Ch_neutral=getattr(amip_cfg, 'louis_Ch_neutral', 1.5e-3),
            louis_Cd_neutral=getattr(amip_cfg, 'louis_Cd_neutral', 1.5e-3),
            # Default from the live field default (NOT a re-typed literal): the
            # truncated 6.283185307e-5 that used to sit here is ~3e-11 off the
            # exact 2*pi/100 km, which gwd_config_for now overlays onto the
            # kernel — a legacy-checkpoint upconvert would silently run a
            # different default k_wave (codex round 1, finding 6).
            mcfarlane_k_wave=getattr(
                amip_cfg, 'mcfarlane_k_wave',
                ExperimentConfig._field_defaults['mcfarlane_k_wave']),
            mcfarlane_N_ref=getattr(amip_cfg, 'mcfarlane_N_ref', 0.01),
            mcfarlane_directional_spread=getattr(amip_cfg, 'mcfarlane_directional_spread', 1.0),
            mcfarlane_tau_max=getattr(amip_cfg, 'mcfarlane_tau_max', 10.0),
            sbm_tau_c=amip_cfg.sbm_tau_c,
            sbm_RH_ref=amip_cfg.sbm_RH_ref,
            sbm_cape_threshold=getattr(amip_cfg, 'sbm_cape_threshold', 70.0),
            bechtold_cape_threshold=getattr(
                amip_cfg, 'bechtold_cape_threshold', 70.0),
            bechtold_conv_top_pa=getattr(
                amip_cfg, 'bechtold_conv_top_pa', 15000.0),
            bechtold_downdraft_evap=getattr(
                amip_cfg, 'bechtold_downdraft_evap', 0.05),
            bechtold_downdraft_alpha=getattr(
                amip_cfg, 'bechtold_downdraft_alpha', 0.3),
            bechtold_downdraft_rh_min=getattr(
                amip_cfg, 'bechtold_downdraft_rh_min', 0.2),
            bechtold_downdraft_transport=getattr(
                amip_cfg, 'bechtold_downdraft_transport', False),
            bechtold_downdraft_entrain_rate=getattr(
                amip_cfg, 'bechtold_downdraft_entrain_rate', 3.0e-4),
            bechtold_downdraft_detrain_scale_m=getattr(
                amip_cfg, 'bechtold_downdraft_detrain_scale_m', 700.0),
            # Missing-field fallback = True (the scheme default): a legacy
            # flat config predating the field must get the SAME closure a
            # fresh default run gets, not silently pin the old one.
            bechtold_use_ifs_cape_closure=getattr(
                amip_cfg, 'bechtold_use_ifs_cape_closure', True),
            bechtold_use_ifs_subcloud_evap=getattr(
                amip_cfg, 'bechtold_use_ifs_subcloud_evap', True),
            bechtold_use_ifs_inplume_precip=getattr(
                amip_cfg, 'bechtold_use_ifs_inplume_precip', True),
            bechtold_dx_m=getattr(amip_cfg, 'bechtold_dx_m', 0.0),
            bechtold_epsilon_deep=getattr(amip_cfg, 'bechtold_epsilon_deep', 1.75e-3),
            bechtold_delta_deep=getattr(amip_cfg, 'bechtold_delta_deep', 0.75e-4),
            bechtold_capdcycl_land_tau_scale=getattr(
                amip_cfg, 'bechtold_capdcycl_land_tau_scale', 1.0),
            bechtold_subcloud_evap_scale=getattr(amip_cfg, 'bechtold_subcloud_evap_scale', 1.0),
            bechtold_rhebc_land=getattr(amip_cfg, 'bechtold_rhebc_land', 0.75),
            bechtold_rhebc_land_deep=getattr(amip_cfg, 'bechtold_rhebc_land_deep', 0.70),
            bechtold_rprcon=getattr(amip_cfg, 'bechtold_rprcon', 1.4e-3),
            bechtold_M_b_max=getattr(amip_cfg, 'bechtold_M_b_max', 0.05),
            bechtold_enable_cmt=getattr(amip_cfg, 'bechtold_enable_cmt', None),
            bechtold_dnoprc=getattr(amip_cfg, 'bechtold_dnoprc', 3.0e-4),
            bechtold_subsidence_solve=getattr(amip_cfg, 'bechtold_subsidence_solve', "implicit_flux"),
            bechtold_rain_vapor_sink=getattr(amip_cfg, 'bechtold_rain_vapor_sink', "formation"),
            convective_buoyancy_death_memory=getattr(amip_cfg, 'convective_buoyancy_death_memory', False),
            convective_cloud=getattr(amip_cfg, 'convective_cloud', False),
            bechtold_use_ifs_downdraft=getattr(
                amip_cfg, 'bechtold_use_ifs_downdraft', True),
            bechtold_use_ifs_shallow_closure=getattr(
                amip_cfg, 'bechtold_use_ifs_shallow_closure', False),
            bechtold_use_ifs_capdcycl=getattr(
                amip_cfg, 'bechtold_use_ifs_capdcycl', True),
            bechtold_use_ifs_land_rhebc=getattr(
                amip_cfg, 'bechtold_use_ifs_land_rhebc', True),
            bechtold_use_ifs_snow_melt=getattr(
                amip_cfg, 'bechtold_use_ifs_snow_melt', True),
            # Convective precip split family — copy through AMIP/checkpoint
            # restore so the physical autoconversion isn't dropped to defaults
            # (codex MED; convective_precip_efficiency was a pre-existing gap).
            convective_precip_efficiency=getattr(
                amip_cfg, 'convective_precip_efficiency', None),
            convective_precip_split=getattr(
                amip_cfg, 'convective_precip_split', 'constant'),
            autoconv_q_c_crit=getattr(amip_cfg, 'autoconv_q_c_crit', 5.0e-4),
            autoconv_pe_max=getattr(amip_cfg, 'autoconv_pe_max', 0.9),
            sigma_b=amip_cfg.sigma_b,
            k_BL_max_per_day=amip_cfg.k_BL_max_per_day,
            k_free_per_day=amip_cfg.k_free_per_day,
            sponge_enabled=getattr(amip_cfg, 'sponge_enabled', False),
            sponge_coeff_per_day=getattr(amip_cfg, 'sponge_coeff_per_day', 2.0),
            sponge_sigma_top=getattr(amip_cfg, 'sponge_sigma_top', 0.15),
            held_suarez_forcing=getattr(amip_cfg, 'held_suarez_forcing', False),
            physics_parameterization=getattr(
                amip_cfg, 'physics_parameterization', 'none',
            ),
            physics_parameterization_checkpoint=getattr(
                amip_cfg, 'physics_parameterization_checkpoint', '',
            ),
            physics_parameterization_stats=getattr(
                amip_cfg, 'physics_parameterization_stats', '',
            ),
            physics_parameterization_hidden_dim=getattr(
                amip_cfg, 'physics_parameterization_hidden_dim', 128,
            ),
            physics_parameterization_layers=getattr(
                amip_cfg, 'physics_parameterization_layers', 3,
            ),
            physics_parameterization_seed=getattr(
                amip_cfg, 'physics_parameterization_seed', 0,
            ),
            distributed=amip_cfg.distributed,
            ensemble_size=amip_cfg.ensemble_size,
        )

    def to_amip_config(self):
        """Convert to AMIPExperimentConfig for backward-compatible serialization.

        Used at serialization boundaries (checkpoint save, legacy config
        export) — not in core runtime paths.
        """
        from legoesm.forcing.amip_config import AMIPExperimentConfig
        _amip_kwargs = dict(
            resolution=self.grid.resolution,
            nlev=self.grid.nlev,
            dt=self.dycore.dt,
            vertical_coord=self.grid.vertical_coord,
            p_top_Pa=self.grid.p_top_Pa,
            stretching=self.grid.stretching,
            tropopause_refine=self.grid.tropopause_refine,
            start_day=self.start_day,
            days=self.days,
            diag_days=self.output.diag_days,
            checkpoint_days=self.output.checkpoint_days,
            dataset=self.dataset,
            forcing_path=self.forcing_path,
            sic_path=self.sic_path,
            sst_var=self.sst_var,
            sic_var=self.sic_var,
            time_var=self.time_var,
            lat_var=self.lat_var,
            lon_var=self.lon_var,
            sst_offset=self.sst_offset,
            sic_scale=self.sic_scale,
            radiation=self.radiation,
            rad_update_steps=self.rad_update_steps,
            physics_update_steps=self.physics_update_steps,
            cld_macmic_num_steps=self.cld_macmic_num_steps,
            morrison_sed_cfl_substeps=self.morrison_sed_cfl_substeps,
            morrison_sed_cfl_substeps_max=self.morrison_sed_cfl_substeps_max,
            morrison_sed_cfl_substeps_strict=self.morrison_sed_cfl_substeps_strict,
            diurnal_cycle=self.diurnal_cycle,
            co2_ppmv=self.co2_ppmv,
            ch4_ppbv=self.ch4_ppbv,
            n2o_ppbv=self.n2o_ppbv,
            S_0=self.S_0,
            ozone_source=self.ozone_source,
            ozone_forcing=self.ozone_forcing,
            ozone_file=self.ozone_file,
            ghg_forcing=self.ghg_forcing,
            ghg_file=self.ghg_file,
            solar_source=self.solar_source,
            solar_file=self.solar_file,
            solar_tsi_var=self.solar_tsi_var,
            solar_spectral_var=self.solar_spectral_var,
            solar_spectral_band_order=self.solar_spectral_band_order,
            aerosol_forcing=self.aerosol_forcing,
            aerosol_file=self.aerosol_file,
            aerosol_reference_aod=self.aerosol_reference_aod,
            volcanic_aerosol_file=self.volcanic_aerosol_file,
            volcanic_aerosol_scale=self.volcanic_aerosol_scale,
            cloud_scheme=self.cloud_scheme,
            microphysics=self.microphysics,
            convection=self.convection,
            turbulence=self.turbulence,
            gravity_wave_drag=self.gravity_wave_drag,
            fix_moisture=self.fix_moisture,
            energy_consistent_moisture_clip=self.energy_consistent_moisture_clip,
            topography=self.topography,
            topo_smoothing=self.topo_smoothing,
            topo_edge_blend=self.topo_edge_blend,
            T_init=self.T_init,
            rh_init=self.rh_init,
            dynamic_albedo=self.dynamic_albedo,
            carbon_cycle=self.carbon_cycle,
            experiment=self.experiment,
            start_year=self.start_year,
            C_H=self.C_H,
            C_E=self.C_E,
            T_ice=self.T_ice,
            albedo_ice=self.albedo_ice,
            albedo_ocean=self.albedo_ocean,
            sfc_emissivity=self.sfc_emissivity,
            emissivity_ice=self.emissivity_ice,
            tau_equator=self.tau_equator,
            tau_pole=self.tau_pole,
            tau_moist_coeff=getattr(self, 'tau_moist_coeff', 0.0115),
            linear_frac=getattr(self, 'linear_frac', 0.2),
            lw_diff_factor=getattr(self, 'lw_diff_factor', 1.66),
            sw_tau_0=getattr(self, 'sw_tau_0', 0.22),
            sw_exponent=getattr(self, 'sw_exponent', 2.0),
            sundqvist_auto_rate=getattr(self, 'sundqvist_auto_rate', 1e-3),
            sundqvist_evap_coeff=getattr(self, 'sundqvist_evap_coeff', 5e-4),
            cloud_rh_crit=getattr(self, 'cloud_rh_crit', 0.7),
            cloud_r_eff_liq=getattr(self, 'cloud_r_eff_liq', 10.0e-6),
            sundqvist_sigmoid_sharpness=getattr(self, 'sundqvist_sigmoid_sharpness', 20.0),
            sbm_T_min_convect=getattr(self, 'sbm_T_min_convect', 200.0),
            louis_l_mix_max=getattr(self, 'louis_l_mix_max', 100.0),
            louis_Ck=getattr(self, 'louis_Ck', 0.4),
            louis_Ri_crit=getattr(self, 'louis_Ri_crit', 0.25),
            louis_b_louis=getattr(self, 'louis_b_louis', 5.0),
            louis_c_louis=getattr(self, 'louis_c_louis', 16.6),
            louis_d_louis=getattr(self, 'louis_d_louis', 5.0),
            louis_z0=getattr(self, 'louis_z0', 1.0e-4),
            louis_Ch_neutral=getattr(self, 'louis_Ch_neutral', 1.5e-3),
            louis_Cd_neutral=getattr(self, 'louis_Cd_neutral', 1.5e-3),
            mcfarlane_k_wave=getattr(
                self, 'mcfarlane_k_wave',
                ExperimentConfig._field_defaults['mcfarlane_k_wave']),
            mcfarlane_N_ref=getattr(self, 'mcfarlane_N_ref', 0.01),
            mcfarlane_directional_spread=getattr(self, 'mcfarlane_directional_spread', 1.0),
            mcfarlane_tau_max=getattr(self, 'mcfarlane_tau_max', 10.0),
            sbm_tau_c=self.sbm_tau_c,
            sbm_RH_ref=self.sbm_RH_ref,
            sbm_cape_threshold=self.sbm_cape_threshold,
            bechtold_cape_threshold=self.bechtold_cape_threshold,
            convective_precip_efficiency=self.convective_precip_efficiency,
            convective_precip_split=self.convective_precip_split,
            autoconv_q_c_crit=self.autoconv_q_c_crit,
            autoconv_pe_max=self.autoconv_pe_max,
            bechtold_conv_top_pa=self.bechtold_conv_top_pa,
            bechtold_downdraft_evap=self.bechtold_downdraft_evap,
            bechtold_downdraft_alpha=self.bechtold_downdraft_alpha,
            bechtold_downdraft_rh_min=self.bechtold_downdraft_rh_min,
            bechtold_use_ifs_cape_closure=self.bechtold_use_ifs_cape_closure,
            bechtold_use_ifs_subcloud_evap=self.bechtold_use_ifs_subcloud_evap,
            bechtold_use_ifs_inplume_precip=self.bechtold_use_ifs_inplume_precip,
            bechtold_dx_m=self.bechtold_dx_m,
            bechtold_epsilon_deep=self.bechtold_epsilon_deep,
            bechtold_delta_deep=self.bechtold_delta_deep,
            bechtold_capdcycl_land_tau_scale=self.bechtold_capdcycl_land_tau_scale,
            bechtold_subcloud_evap_scale=self.bechtold_subcloud_evap_scale,
            bechtold_rhebc_land=self.bechtold_rhebc_land,
            bechtold_rhebc_land_deep=self.bechtold_rhebc_land_deep,
            bechtold_rprcon=self.bechtold_rprcon,
            bechtold_M_b_max=self.bechtold_M_b_max,
            bechtold_enable_cmt=self.bechtold_enable_cmt,
            bechtold_dnoprc=self.bechtold_dnoprc,
            bechtold_downdraft_entrain_rate=self.bechtold_downdraft_entrain_rate,
            bechtold_downdraft_detrain_scale_m=self.bechtold_downdraft_detrain_scale_m,
            bechtold_downdraft_transport=self.bechtold_downdraft_transport,
            bechtold_subsidence_solve=self.bechtold_subsidence_solve,
            bechtold_rain_vapor_sink=self.bechtold_rain_vapor_sink,
            convective_buoyancy_death_memory=self.convective_buoyancy_death_memory,
            convective_cloud=self.convective_cloud,
            bechtold_use_ifs_downdraft=self.bechtold_use_ifs_downdraft,
            bechtold_use_ifs_shallow_closure=self.bechtold_use_ifs_shallow_closure,
            bechtold_use_ifs_capdcycl=self.bechtold_use_ifs_capdcycl,
            bechtold_use_ifs_land_rhebc=self.bechtold_use_ifs_land_rhebc,
            bechtold_use_ifs_snow_melt=self.bechtold_use_ifs_snow_melt,
            sigma_b=self.sigma_b,
            k_BL_max_per_day=self.k_BL_max_per_day,
            k_free_per_day=self.k_free_per_day,
            held_suarez_forcing=self.held_suarez_forcing,
            physics_parameterization=self.physics_parameterization,
            physics_parameterization_checkpoint=self.physics_parameterization_checkpoint,
            physics_parameterization_stats=self.physics_parameterization_stats,
            physics_parameterization_hidden_dim=self.physics_parameterization_hidden_dim,
            physics_parameterization_layers=self.physics_parameterization_layers,
            physics_parameterization_seed=self.physics_parameterization_seed,
            monthly_means=self.output.monthly_means,
            cmip_output=self.output.cmip_output,
            clear_sky_diag=self.output.clear_sky_diag,
            distributed=self.distributed,
            ensemble_size=self.ensemble_size,
            output_dir=self.output.output_dir,
        )
        # Filter to the legacy AMIP schema's fields — ExperimentConfig has
        # accreted many newer knobs the flat AMIPExperimentConfig never
        # mirrored; drop those instead of raising (drift-proof round-trip).
        return AMIPExperimentConfig(**{
            k: v for k, v in _amip_kwargs.items()
            if k in AMIPExperimentConfig._fields})


# ======================================================================
# Native JSON serialization for ExperimentConfig
# ======================================================================

_SUB_CONFIGS = {
    "grid": GridConfig,
    "dycore": DycoreConfig,
    "output": OutputConfig,
}


def experiment_config_to_dict(config: ExperimentConfig) -> dict:
    """Serialize ExperimentConfig to a JSON-safe dict.

    Sub-configs (grid, dycore, output) are inlined as nested dicts. This
    is the canonical serialization format. ``output.evaluation`` is
    itself a nested ``EvaluationConfig`` NamedTuple one level deeper —
    without this it round-trips through ``json.dumps`` as a bare
    positional list (NamedTuple is a tuple), losing field names.
    """
    d = config._asdict()
    # ``turbulence_override`` is a RUNTIME-ONLY injection (it may carry a
    # per-column JAX array C_K) — it is NOT persisted: the serialized config is
    # the base config, and the override is re-applied in memory after load (the
    # correction loop's build_driver). Drop it so it cannot be stringified +
    # silently corrupted on round-trip.
    d["turbulence_override"] = None
    for key in _SUB_CONFIGS:
        sub = d[key]
        if hasattr(sub, '_asdict'):
            sub_d = sub._asdict()
            if hasattr(sub_d.get('evaluation'), '_asdict'):
                sub_d['evaluation'] = sub_d['evaluation']._asdict()
            d[key] = sub_d
    return d


def experiment_config_from_dict(d: dict, *, strict: bool = False) -> ExperimentConfig:
    """Reconstruct ExperimentConfig from a dict (e.g. loaded from JSON).

    Unknown fields are silently dropped for forward-compatibility
    (so older checkpoints with removed fields still load).

    ``strict=True`` raises on any unknown key instead. Use it for LAUNCH
    configs: silently dropping a key there means a typo -- e.g. ``grid.type``
    where the field is ``grid_type`` -- loads cleanly and runs a DIFFERENT
    experiment than the file describes, with nothing in the log to say so.
    Checkpoint reload keeps the permissive default.
    """
    if strict:
        unknown = []
        for key, cls in _SUB_CONFIGS.items():
            if key in d and isinstance(d[key], dict):
                known_sub = set(cls._fields)
                unknown += [f"{key}.{k}" for k in d[key] if k not in known_sub
                            and not (key == "output" and k == "evaluation")]
                ev = d[key].get("evaluation")
                if isinstance(ev, dict):
                    unknown += [f"{key}.evaluation.{k}" for k in ev
                                if k not in set(EvaluationConfig._fields)]
        known_top = set(ExperimentConfig._fields)
        unknown += [k for k in d if k not in known_top]
        if unknown:
            raise ValueError(
                "unknown config field(s): " + ", ".join(sorted(unknown))
                + ". Loading a launch config with strict=True refuses to "
                  "silently drop keys, because a dropped key runs a different "
                  "experiment than the file describes.")
    # Reconstruct sub-configs
    sub_values = {}
    for key, cls in _SUB_CONFIGS.items():
        if key in d and isinstance(d[key], dict):
            sub_d = dict(d[key])
            if isinstance(sub_d.get('evaluation'), dict):
                known_eval = set(EvaluationConfig._fields)
                filtered_eval = {
                    k: v for k, v in sub_d['evaluation'].items() if k in known_eval
                }
                # ``suites`` is a tuple field; JSON round-trips it as a list, so
                # coerce back so the reconstructed config == the original.
                if isinstance(filtered_eval.get('suites'), list):
                    filtered_eval['suites'] = tuple(filtered_eval['suites'])
                sub_d['evaluation'] = EvaluationConfig(**filtered_eval)
            known_sub = set(cls._fields)
            filtered = {k: v for k, v in sub_d.items() if k in known_sub}
            sub_values[key] = cls(**filtered)

    # Filter top-level fields
    known = set(ExperimentConfig._fields)
    filtered = {k: v for k, v in d.items() if k in known}
    filtered.update(sub_values)

    # Legacy migration: ``morrison_dep_coeff`` was declared 1e-8 (1e5 OFF the
    # real MorrisonConfig.dep_coeff=1e-3) while UNWIRED, so every serialized
    # config from that era carries the inert 1e-8.  Now that the scalar is
    # wired, loading it verbatim would (a) retune Morrison's deposition 1e5
    # DOWN, or (b) raise on non-Morrison schemes — neither is what the old
    # run did (nothing).  Rewriting the old default to the new one preserves
    # the ACTUAL old behaviour (leaf stayed 1e-3).  Any other stored value is
    # kept: it was a deliberate (if then-inert) user choice.
    # Schema-version gate (codex 2026-07-26 round 3, item 2): dicts written
    # by the CURRENT codec carry ``config_schema_version`` >= 2 and are NOT
    # migrated — a fresh JSON deliberately carrying 1e-8 is loaded verbatim
    # and then REFUSED by validate_strict's bounds.  Only unmarked (legacy)
    # dicts, which predate the wiring and where 1e-8 was inert, migrate.
    _legacy = int(d.get("config_schema_version", 1)) < 2
    if _legacy and filtered.get("morrison_dep_coeff") == 1e-8:
        import warnings
        warnings.warn(
            "legacy config carries morrison_dep_coeff=1e-8 — the old INERT "
            "declaration default, which is also OUTSIDE the valid range "
            "[1e-4, 1e-2] now that the scalar is wired (validate_strict "
            "would refuse it), so no modern config can mean it. Migrating "
            "to 1e-3, the value the legacy run actually used.",
            stacklevel=2,
        )
        filtered["morrison_dep_coeff"] = 1e-3

    return ExperimentConfig(**filtered)


def config_to_dict(config) -> dict:
    """Generic config -> JSON-safe dict codec (canonical home).

    Accepts an ``ExperimentConfig`` or the legacy flat ``AMIPExperimentConfig``;
    nested sub-config NamedTuples (grid/dycore/output) are inlined as dicts.
    This is the codec the experiment-level checkpoint/restart I/O in
    ``driver`` uses, so that layer no longer reaches up into
    ``forcing.amip_config`` for serialization (federation carve, Step 3).
    """
    d = config._asdict()
    # Runtime-only injection — never serialized (see experiment_config_to_dict).
    if "turbulence_override" in d:
        d["turbulence_override"] = None
    for key, val in d.items():
        if hasattr(val, "_asdict"):
            d[key] = val._asdict()
    # Schema marker: dicts written by the current codec are exempt from
    # legacy migrations on load (see experiment_config_from_dict — v2 gates
    # the morrison_dep_coeff 1e-8 rewrite to UNMARKED legacy dicts, so a
    # fresh JSON deliberately carrying an out-of-bounds value is refused by
    # validate_strict instead of silently rewritten).  Unknown keys are
    # dropped by the loader's field filter, so this is forward-compatible.
    d["config_schema_version"] = 2
    return d


def config_from_dict(d: dict):
    """Reconstruct a config from a dict written by :func:`config_to_dict`.

    Auto-detects the schema: an ``ExperimentConfig`` serializes with nested
    ``grid``/``dycore`` sub-config dicts (and fields like ``seed`` that the AMIP
    schema lacks), so it round-trips through ``experiment_config_from_dict``;
    otherwise the dict is the legacy flat ``AMIPExperimentConfig``.  The AMIP
    type is imported lazily so this module needs no top-level ``forcing`` import.
    """
    if isinstance(d.get("grid"), dict) or isinstance(d.get("dycore"), dict):
        return experiment_config_from_dict(d)
    from legoesm.forcing.amip_config import AMIPExperimentConfig

    known = set(AMIPExperimentConfig._fields)
    return AMIPExperimentConfig(**{k: v for k, v in d.items() if k in known})


def save_experiment_config(config: ExperimentConfig, path: Path | str) -> None:
    """Save ExperimentConfig to JSON file."""
    with open(path, "w") as f:
        json.dump(experiment_config_to_dict(config), f, indent=2, default=str)


def load_experiment_config(path: Path | str, *,
                          strict: bool = False) -> ExperimentConfig:
    """Load ExperimentConfig from JSON file.

    ``strict=True`` refuses unknown keys -- see
    :func:`experiment_config_from_dict`. Launch scripts should pass it.
    """
    with open(path) as f:
        return experiment_config_from_dict(json.load(f), strict=strict)
