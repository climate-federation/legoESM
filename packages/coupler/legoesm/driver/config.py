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
    vertical_coord: str = "hybrid"   # sigma, hybrid
    p_top_Pa: float = 200.0
    stretching: float = 2.0
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

    # #1029: lat-lon C-grid PE energy-consistency options (threaded into
    # ``CGridLatLonPrimitiveEquationConfig`` by ``component_factory``).
    # ``energy_paired_conversion=True`` computes the κT v·∇ln p part of the
    # adiabatic conversion as the face-averaged product u·pg_corr/c_p —
    # discretely adjoint to the momentum PGF-correction work.  The legacy
    # product-of-cell-averages (False) is a spurious energy source over
    # steep terrain ridges: the DCMIP 2-0-0 rest state grows at 2.6
    # e-folds/day (the AMIP latlon Andes lid-wave killer; ablation probe
    # 26276143).  ``pgf_scheme`` selects the momentum PGF discretisation
    # ("two_term" legacy | "lin1997" FV3-faithful cross-product).
    energy_paired_conversion: bool = False
    pgf_scheme: str = "two_term"

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
    # Shapiro-form per-step strength of the same vertical del4 operator
    # (fraction of the 2Δσ mode removed per step, unconditionally stable in
    # (0,1]).  The ERA5-IC MPAS lane needs ~0.5 (the physics-forced
    # checkerboard outgrows the explicit rate form's stability-limited
    # damping at production dt); default 0.0 = off.
    mpas_vert4_t_filter: float = 0.0
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
    diag_days: int = 5
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

VALID_CLOUD_SCHEMES = ("none", "sundqvist", "xu_randall", "resolved")

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
    rejects -- "none+hines", "e3sm_cam+hines", "ml_emulator+hines", duplicates
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
        ExperimentConfig(gravity_wave_drag=value).validate_strict()
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
VALID_SURFACE_BULK = ("constant", "coare3", "large_yeager")


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
    cloud_rh_crit_bl: float = 0.7
    cloud_sigma_bl: float = 1.0
    # Route a moist higher-order turbulence closure's (CLUBB) sub-grid PDF cloud
    # fraction into the cloud optics instead of the RH grid-scale one — the
    # marine-Sc over-bright albedo lever.  Maps to
    # ``RadiationConfig.use_clubb_cloud_fraction``; requires diagnostic CLUBB
    # turbulence (turbulence='clubb').  False (default) is byte-identical.
    use_clubb_cloud_fraction: bool = False
    # Opt-in convective (cumulus) cloud-fraction source (Slingo 1987).  The
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
    #                          LW-active).  Bounds (5e-5, 1e-3).
    #   cloud_conv_cloud_max — convective (Slingo) cover cap.  Bounds (0.1, 1.0).
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
    #   cloud_p_xr / cloud_alpha_xr — Xu-Randall cloud-fraction sensitivity
    #   knobs; HIGHER p_xr / LOWER alpha_xr => fraction stays fractional as
    #   moisture rises (flattens the overcast runaway).
    cloud_p_xr: float | None = None
    cloud_alpha_xr: float | None = None
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
    # stable functions ("dyer1974" default = byte-identical -5*zeta;
    # "grachev2007_sheba"/"gryanik2020" = SHEBA Arctic forms;
    # "beljaars_holtslag1991").
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
    land_surface_scheme: str = "simple_seb"
    # Initial multilayer soil water as a fraction of saturation (theta_init =
    # frac * theta_sat) for the cold-start (#730). Default 0.5 is byte-identical to
    # the init_multilayer_land_state default. The multilayer over-evaporation wet
    # loop is precip-recycling-driven (land P ~= land ET), so a DRIER start (e.g.
    # 0.25) can tip the land into the slab-like dry attractor (less ET -> less low
    # cloud -> warmer land) instead of the cold-cloudy wet attractor. Only affects
    # use_multilayer_land runs.
    land_soil_moisture_init_frac: float = 0.5
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
    # Extra truly-diffusive (unanchored) smoothing passes applied after the
    # anchored ``topo_smoothing`` passes.  The anchored smoother SATURATES
    # (re-blends with the original field each pass, so values beyond ~4 are
    # a no-op); these passes keep removing grid-scale terrain power.  #1029:
    # 4 passes eliminate the episodic mountain-wave breaking blowup of the
    # coarse lat-lon lane over real ETOPO terrain (~84% Tibet peak retained
    # at 24x48).  0 = bit-identical legacy topography.
    topo_diffusive_smoothing: int = 0
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
    multilayer_n_layers: int = 10        # soil discretization
    multilayer_soil_depth: float = 3.0   # m
    multilayer_soil_texture: str = "loam"  # van-Genuchten preset
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
    # outbound internet, so stage the file and set this).  Ignored unless
    # use_multilayer_land is True.
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
    # grid-mean cloud water q_c is split into liquid + ice via
    # f_ice(T) = clip((T_warm - T) / (T_warm - T_cold), 0, 1), and the
    # ice portion is fed to RRTMGP so cirrus has radiative effect.  This
    # is NOT a prognostic ice scheme (no Bergeron, no sedimentation —
    # just a radiative diagnostic; see Phase 2 of cloud-micro plan).
    #   cloud_T_warm  : liquid-only above this T [K] (default freezing pt)
    #   cloud_T_cold  : ice-only below this T  [K] (default homog. ice
    #                   nucleation, -38 C)
    #   cloud_r_eff_ice : ice effective radius [m]
    cloud_T_warm: float = constants.T_freeze
    cloud_T_cold: float = 235.15
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
    albedo_ice: float = 0.65
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
    mcfarlane_k_wave: float = 6.283185307e-5    # McFarlaneConfig.k_wave [1/m]
    mcfarlane_N_ref: float = 0.01               # McFarlaneConfig.N_ref [1/s]
    mcfarlane_directional_spread: float = 1.0   # McFarlaneConfig.directional_spread
    mcfarlane_tau_max: float = 10.0             # McFarlaneConfig.tau_max [Pa]
    # Morrison ice-microphysics tunables (active when microphysics='morrison'
    # — silently filtered out by hasattr when Sundqvist/Kessler is active).
    morrison_bergeron_rate: float = 1e-3        # MorrisonConfig.bergeron_rate [1/s]
    morrison_rime_coeff: float = 1.0            # MorrisonConfig.rime_coeff
    morrison_dep_coeff: float = 1e-8            # MorrisonConfig.dep_coeff
                                                 # (Morrison-2005 q_i^(1/3)·N_i^(2/3) form)
    morrison_agg_coeff: float = 1e-3            # MorrisonConfig.agg_coeff [1/s]
    morrison_k_au: float = 6e2                  # MorrisonConfig.k_au [1/(kg*s)]
    # SBM convective precip efficiency: fraction of column-net drying that
    # precipitates directly as rain (rest is detrained as condensate).
    sbm_precip_efficiency: float = 0.5          # SBMConfig.precip_efficiency
    # Morrison phase-aware saturation_adjustment (P7).  Default False to
    # preserve legacy calibration baselines; set True to let condensation
    # target q_sat_ice at T < phase_T_cold (235 K), routing cold-T
    # condensate directly to q_i with the matching L_s release.
    morrison_phase_aware_sat_adj: bool = False
    # Cirrus cf branch (CloudConfig RH_i ramp).  Linear from cf=0 at
    # rh_ice_crit to cf=1 at rh_ice_sat; max with warm Sundqvist cf.
    cloud_rh_ice_crit: float = 0.95             # CloudConfig.rh_ice_crit
    cloud_rh_ice_sat: float = 1.30              # CloudConfig.rh_ice_sat
    # Bechtold deep-convection CAPE trigger threshold [J/kg].  Deep convection
    # fires only above this CAPE; lowering it lets convection trigger more
    # readily at coarse resolution (where CAPE is under-resolved), which is
    # the lever for the AMIP convective-precipitation deficit.  Default matches
    # BechtoldConfig.cape_threshold (byte-identical when unset).
    bechtold_cape_threshold: float = 70.0
    # Remaining spec'd Bechtold tunables threaded through the pipeline
    # (#869 polar-night campaign levers): cloud-base mass-flux stability cap
    # [kg/m^2/s] and the Gregory-1997 CMT coefficients.  Defaults match
    # BechtoldConfig (byte-identical when unset).  M_b_max's default 0.02
    # sits at the BOTTOM of its __param_spec__ bounds (0.02-0.15) and the
    # mass flux runs pinned there — a first-class tuning lever.
    bechtold_m_b_max: float = 0.02
    bechtold_cmt_c_u: float = 0.7
    bechtold_cmt_c_d: float = 0.7
    # Quasi-equilibrium heating-ceiling ratio (BechtoldConfig.
    # cape_sink_heating_ratio; the C12 warm-runaway sink lever).
    bechtold_cape_sink_heating_ratio: float = 5.0
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
    # MPAS-standalone RRTMGP optics precision (the coupled pipeline runs
    # fp64 optics): True = fp32 tables+RTE (perf default), False = fp64
    # (parity with cube/latlon; the ERA5-IC lane's extreme Antarctic
    # columns are a suspected fp32-optics NaN trigger).
    mpas_rrtmgp_fp32: bool = True

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

    def validate_strict(self) -> None:
        """Raise ValueError for invalid parameter values.

        Called before simulation start to catch configuration errors
        early, before JIT compilation or data loading.
        """
        g = self.grid
        d = self.dycore
        errors: list[str] = []
        if g.resolution <= 0:
            errors.append(f"grid.resolution must be > 0, got {g.resolution}")
        if g.nlev <= 0:
            errors.append(f"grid.nlev must be > 0, got {g.nlev}")
        if g.p_top_Pa <= 0:
            errors.append(f"grid.p_top_Pa must be > 0, got {g.p_top_Pa}")
        if d.dt <= 0:
            errors.append(f"dycore.dt must be > 0, got {d.dt}")
        if d.hyperdiff_scale < 0:
            errors.append(f"dycore.hyperdiff_scale must be >= 0, got {d.hyperdiff_scale}")
        if d.div_damp_scale < 0:
            errors.append(f"dycore.div_damp_scale must be >= 0, got {d.div_damp_scale}")
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
        if d.discretization not in DISCRETIZATION_OPTIONS:
            errors.append(
                f"dycore.discretization must be one of {DISCRETIZATION_OPTIONS}, "
                f"got {d.discretization!r}"
            )
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
        if self.bechtold_m_b_max <= 0:
            errors.append(
                f"bechtold_m_b_max must be > 0, got {self.bechtold_m_b_max}"
            )
        if self.bechtold_cmt_c_u < 0:
            errors.append(
                f"bechtold_cmt_c_u must be >= 0, got {self.bechtold_cmt_c_u}"
            )
        if self.bechtold_cmt_c_d < 0:
            errors.append(
                f"bechtold_cmt_c_d must be >= 0, got {self.bechtold_cmt_c_d}"
            )
        if self.bechtold_cape_sink_heating_ratio <= 0:
            errors.append(
                f"bechtold_cape_sink_heating_ratio must be > 0, got "
                f"{self.bechtold_cape_sink_heating_ratio}"
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
        # use_clubb_cloud_fraction is enforced (turbulence must be clubb) only
        # inside build_physics_pipeline, which the mpas/spectral standalone
        # radiation paths never build — so the opt-in would silently no-op
        # there.  Reject it loudly on those backends (dispatch-hardening,
        # mirrors the condensate-scheme guard above).
        if (self.use_clubb_cloud_fraction
                and self.dycore.discretization
                in _NO_CLOUD_THREAD_DISCRETIZATIONS):
            errors.append(
                "use_clubb_cloud_fraction=True is not wired into the "
                f"{self.dycore.discretization!r} radiation path (that backend "
                "builds RadiationConfig directly, bypassing the shared physics "
                "pipeline that enforces it); it would silently no-op.  Use a "
                "pipeline backend (cd-grid / latlon) with turbulence='clubb', "
                "or drop --use-clubb-cloud-fraction."
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
        if self.surface_stability_scheme not in _valid_stability:
            errors.append(
                f"surface_stability_scheme must be one of {_valid_stability}, "
                f"got {self.surface_stability_scheme!r}"
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
        if (self.surface_tiled and self.use_multilayer_land
                and not self.slab_land_active and not self.land_mask_path
                and self.topography == "flat"):
            errors.append(
                "use_multilayer_land + surface_tiled with topography='flat' and no "
                "land-mask file has NO land (elevation-derived f_land is 0 "
                "everywhere) — pass a real --topography or a --land-mask-file."
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
        # Soil-moisture init fraction of saturation: finite, in (0, 1].
        if not (0.0 < self.land_soil_moisture_init_frac <= 1.0):
            errors.append(
                f"land_soil_moisture_init_frac (theta_init/theta_sat) must be "
                f"finite and in (0, 1]; got {self.land_soil_moisture_init_frac!r}."
            )
        # Land surface-scheme membership (mirror the model_driver dispatch so a
        # typo fails here, not at run time).
        _valid_land_surface = ("simple_seb", "two_leaf")
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
        # Optional cloud-tuning override bounds (mirror CloudConfig.__param_spec__
        # so an out-of-range knob fails early, not deep in the cloud diagnosis).
        for _f, _lo, _hi in (
            ("cloud_rh_crit", 0.5, 0.99),
            ("cloud_q_c_diagnostic", 5.0e-5, 1.0e-3),
            ("cloud_conv_cloud_max", 0.1, 1.0),
            ("cloud_conv_cloud_condensate", 1.0e-5, 1.0e-3),
            ("cloud_inhomogeneity_factor", 0.3, 1.0),
            ("cloud_fsd", 0.0, 1.0),
            ("cloud_p_xr", 0.05, 1.0),
            ("cloud_alpha_xr", 10.0, 1000.0),
            ("cloud_adiabatic_lwc_rate", 5.0e-7, 3.0e-6),
        ):
            _v = getattr(self, _f)
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
        _composable_stateless = ("rayleigh", "lindzen", "mcfarlane", "hines")
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
        )
        dycore = DycoreConfig(
            dt=amip_cfg.dt,
            hyperdiff_scale=getattr(amip_cfg, 'hyperdiff_scale', 1.0),
        )
        output = OutputConfig(
            output_dir=getattr(amip_cfg, 'output_dir', ''),
            diag_days=amip_cfg.diag_days,
            checkpoint_days=amip_cfg.checkpoint_days,
            monthly_means=getattr(amip_cfg, 'monthly_means', False),
            cmip_output=getattr(amip_cfg, 'cmip_output', False),
            clear_sky_diag=getattr(amip_cfg, 'clear_sky_diag', False),
            budget_ledger=getattr(amip_cfg, 'budget_ledger', False),
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
            cloud_rh_crit_bl=getattr(amip_cfg, 'cloud_rh_crit_bl', 0.7),
            cloud_sigma_bl=getattr(amip_cfg, 'cloud_sigma_bl', 1.0),
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
            topo_diffusive_smoothing=getattr(
                amip_cfg, 'topo_diffusive_smoothing', 0),
            land_mask_path=getattr(amip_cfg, 'land_mask_path', ''),
            albedo_land_path=getattr(amip_cfg, 'albedo_land_path', ''),
            albedo_land_month=getattr(amip_cfg, 'albedo_land_month', 0),
            C_land=getattr(amip_cfg, 'C_land', 2.0e5),
            emissivity_land=getattr(amip_cfg, 'emissivity_land', constants.emissivity_land),
            beta_land=getattr(amip_cfg, 'beta_land', 1.0),
            use_multilayer_land=getattr(amip_cfg, 'use_multilayer_land', False),
            multilayer_n_layers=getattr(amip_cfg, 'multilayer_n_layers', 10),
            multilayer_soil_depth=getattr(amip_cfg, 'multilayer_soil_depth', 3.0),
            multilayer_soil_texture=getattr(amip_cfg, 'multilayer_soil_texture', 'loam'),
            era5_land_ic_path=getattr(amip_cfg, 'era5_land_ic_path', ''),
            cloud_T_warm=getattr(amip_cfg, 'cloud_T_warm', constants.T_freeze),
            cloud_T_cold=getattr(amip_cfg, 'cloud_T_cold', 235.15),
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
            mcfarlane_k_wave=getattr(amip_cfg, 'mcfarlane_k_wave', 6.283185307e-5),
            mcfarlane_N_ref=getattr(amip_cfg, 'mcfarlane_N_ref', 0.01),
            mcfarlane_directional_spread=getattr(amip_cfg, 'mcfarlane_directional_spread', 1.0),
            mcfarlane_tau_max=getattr(amip_cfg, 'mcfarlane_tau_max', 10.0),
            sbm_tau_c=amip_cfg.sbm_tau_c,
            sbm_RH_ref=amip_cfg.sbm_RH_ref,
            sbm_cape_threshold=getattr(amip_cfg, 'sbm_cape_threshold', 70.0),
            bechtold_cape_threshold=getattr(
                amip_cfg, 'bechtold_cape_threshold', 70.0),
            bechtold_m_b_max=getattr(amip_cfg, 'bechtold_m_b_max', 0.02),
            bechtold_cmt_c_u=getattr(amip_cfg, 'bechtold_cmt_c_u', 0.7),
            bechtold_cmt_c_d=getattr(amip_cfg, 'bechtold_cmt_c_d', 0.7),
            bechtold_cape_sink_heating_ratio=getattr(
                amip_cfg, 'bechtold_cape_sink_heating_ratio', 5.0),
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
            topo_diffusive_smoothing=self.topo_diffusive_smoothing,
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
            mcfarlane_k_wave=getattr(self, 'mcfarlane_k_wave', 6.283185307e-5),
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
            bechtold_downdraft_transport=self.bechtold_downdraft_transport,
            bechtold_downdraft_entrain_rate=self.bechtold_downdraft_entrain_rate,
            bechtold_downdraft_detrain_scale_m=self.bechtold_downdraft_detrain_scale_m,
            bechtold_use_ifs_cape_closure=self.bechtold_use_ifs_cape_closure,
            bechtold_use_ifs_subcloud_evap=self.bechtold_use_ifs_subcloud_evap,
            bechtold_use_ifs_inplume_precip=self.bechtold_use_ifs_inplume_precip,
            bechtold_dx_m=self.bechtold_dx_m,
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


def experiment_config_from_dict(d: dict) -> ExperimentConfig:
    """Reconstruct ExperimentConfig from a dict (e.g. loaded from JSON).

    Unknown fields are silently dropped for forward-compatibility
    (so older checkpoints with removed fields still load).
    """
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


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    """Load ExperimentConfig from JSON file."""
    with open(path) as f:
        return experiment_config_from_dict(json.load(f))
