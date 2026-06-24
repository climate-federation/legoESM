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
from pathlib import Path
from typing import NamedTuple

from legoesm import constants


# Canonical AIMIP variant set.  Single source of truth — imported by
# ``scripts/run/run_aimip.py`` and the ``validate_strict`` rule below.
# Empty string = not an AIMIP run (preserves backward-compat for
# existing AMIP configs).
AIMIP_VARIANTS: tuple[str, ...] = (
    "", "classical", "column_nn", "sfno_physics", "sfno_full",
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
    # Default ``"ssp_rk3"`` preserves bit-equivalent behaviour for
    # the existing scientific validation suite.  ``"ssp_rk3_scan"``
    # is the opt-in for production at scale.
    #
    # ``"auto"`` (the run_amip CLI default since 2026-06-10) selects
    # each dycore's own stable default in ``component_factory``:
    # cube / lat-lon keep ``ssp_rk3``; MPAS gets ``ssp_rk54_scan``
    # (its biharmonic hyperdiffusion eigenvalues at production dt fall
    # outside the ssp_rk3 stability region — the former "hidden CFL"
    # blow-up); spectral keeps ``ssp_rk54``.  Any explicit scheme name
    # (including ``ssp_rk3`` on MPAS) is forwarded verbatim, so
    # deliberate integrator-sensitivity runs are still possible.
    time_integrator: str = "ssp_rk3"


class OutputConfig(NamedTuple):
    """Output and diagnostics configuration."""
    output_dir: str = ""
    diag_days: int = 5
    checkpoint_days: int = 0
    monthly_means: bool = False
    cmip_output: bool = False
    clear_sky_diag: bool = False
    checkpoint_format: str = "npz"  # npz, zarr
    diagnostics_perf_mode: str = "auto"  # auto, always, never
    cmip_resolution_deg: float = 5.0  # lat-lon grid spacing for CMIP output [degrees]
    # Wallclock-aware mid-job checkpointing for long (century-scale) HPC chains.
    # ``max_wallclock_seconds=0`` disables it (default); set it to the SLURM
    # ``--time`` budget so the run checkpoints and exits cleanly with
    # ``restart_buffer_seconds`` to spare, letting a dependency chain resume.
    max_wallclock_seconds: float = 0.0
    restart_buffer_seconds: float = 600.0


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
    # ``RRTMGPConfig.gpoint_batch_size``).  0 = memory-frugal checkpointed scan
    # (REQUIRED for reverse-mode AD / training).  >0 = process g-points in
    # parallel blocks of this size via vmap — FORWARD/inference only, ~6x faster
    # radiation on GPU; ~16-32 recovers most parallelism while bounding memory.
    rrtmgp_gpoint_batch_size: int = 0
    # G-point checkpointing in the RRTMGP two-stream scan (see
    # ``RRTMGPConfig.gpoint_checkpoint``).  True (default) = ``jax.checkpoint``
    # with ``prevent_cse=True`` per g-point — memory-frugal, REQUIRED for
    # reverse-mode AD / training.  False = plain ``lax.scan`` (no prevent_cse):
    # smaller compiled footprint / faster cold compile for FORWARD/inference
    # runs, used to relieve the XLA-CPU LLVM-JIT code-region pressure.
    rrtmgp_gpoint_checkpoint: bool = True
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
    # These are the SW/LW knob for the coare3 moisture-driven albedo overshoot.
    cloud_rh_crit: float | None = None
    cloud_q_c_diagnostic: float | None = None
    cloud_conv_cloud_max: float | None = None
    microphysics: str = "none"
    # Aerosol-CCN coupling: diagnose the specified cloud-droplet number
    # from the prescribed aerosol optical depth (Andreae 2009 AOT–CCN
    # inversion) instead of the scheme's constant Nc_0.  Requires
    # aerosol_forcing="external" and a specified-Nc double-moment
    # microphysics (morrison); validated in build_physics_pipeline.
    nc_from_aerosol: bool = False

    # Convection / Turbulence / GWD
    convection: str = "sbm"            # sbm, dca, kuo, mass_flux, edmf, none
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
    gravity_wave_drag: str = "none"    # rayleigh, lindzen, mcfarlane, hines, prognostic_spectral, e3sm_cam, ml_emulator, none

    # Conservation
    fix_moisture: bool = False
    # Issue #323: make the per-step ``max(q_v, 0)`` floor on the physics
    # tracer update moist-static-energy-conserving (remove the latent heat
    # of the clipped vapour sink).  Opt-in for the kessler+sbm wind blow-up;
    # default off => bit-identical.
    energy_consistent_moisture_clip: bool = False

    # Topography
    topography: str = "flat"
    topo_smoothing: int = 4
    topo_edge_blend: float = 0.3
    # Optional land-sea-mask NetCDF (CMIP6 sftlf / ERA5 lsm).  When set,
    # the land fraction is taken from this file and the slab-land tile
    # is activated; empty → ocean-only surface.
    land_mask_path: str = ""

    # Surface
    T_init: float = 300.0
    rh_init: float = 0.7
    dynamic_albedo: bool = False
    carbon_cycle: str = "none"

    # Initial conditions
    ic: str = "default"   # "default" (uniform T_init), "standard" (lapse-rate + equator-pole gradient), or "era5"
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
    sfc_emissivity: float = 0.97
    emissivity_ice: float = 0.95
    tau_equator: float = 7.2
    tau_pole: float = 1.8
    sbm_tau_c: float = 7200.0
    sbm_RH_ref: float = 0.7
    sbm_cape_threshold: float = 70.0
    sigma_b: float = 0.7
    k_BL_max_per_day: float = 1.0
    k_free_per_day: float = 0.1

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
            # Milestone-1 limitation (codex review MAJOR): checkpoint and
            # diagnostics writers assume a single process or an mpi4jax
            # topology — under multi-controller SPMD every process would
            # hit the same output path (concurrent clobber) or call
            # device_get on non-fully-addressable global arrays.  Refuse
            # LOUDLY until the gathered root-only writers are wired.
            if self.output.checkpoint_days > 0:
                errors.append(
                    "distributed_mode='spmd' does not support "
                    "checkpointing yet (output.checkpoint_days="
                    f"{self.output.checkpoint_days}); set "
                    "checkpoint_days=0 — gathered root-only restart "
                    "writes are a follow-up"
                )
            if self.output.diag_days > 0:
                errors.append(
                    "distributed_mode='spmd' does not support the "
                    "diagnostics writer yet (output.diag_days="
                    f"{self.output.diag_days}); set diag_days=0 — "
                    "gathered root-only diagnostics are a follow-up"
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
        _valid_radiation = ("none", "gray", "rrtmgp", "rrtmg")
        if self.radiation not in _valid_radiation:
            errors.append(
                f"radiation must be one of {_valid_radiation}, "
                f"got {self.radiation!r}"
            )
        _valid_cloud_schemes = ("none", "sundqvist", "xu_randall", "resolved")
        if self.cloud_scheme not in _valid_cloud_schemes:
            errors.append(
                f"cloud_scheme must be one of {_valid_cloud_schemes}, "
                f"got {self.cloud_scheme!r}"
            )
        _valid_microphysics = (
            "none", "kessler", "sundqvist", "seifert_beheng",
            "morrison", "thompson", "p3", "sdm", "fast_sbm", "ml_emulator",
        )
        if self.microphysics not in _valid_microphysics:
            errors.append(
                f"microphysics must be one of {_valid_microphysics}, "
                f"got {self.microphysics!r}"
            )
        # Physics-scheme membership (mirror the integration.py factory sets so
        # a typo fails here, not only at JIT-compile inside integration.py).
        _valid_convection = (
            "sbm", "dca", "kuo", "mass_flux", "edmf", "zhang_mcfarlane",
            "kain_fritsch", "emanuel", "tiedtke", "bechtold", "none",
        )
        if self.convection not in _valid_convection:
            errors.append(
                f"convection must be one of {_valid_convection}, "
                f"got {self.convection!r}"
            )
        _valid_turbulence = (
            "smagorinsky", "louis", "tke", "mynn25", "clubb_lite", "clubb",
            "holtslag_boville", "ysu", "edmf", "none",
        )
        if self.turbulence not in _valid_turbulence:
            errors.append(
                f"turbulence must be one of {_valid_turbulence}, "
                f"got {self.turbulence!r}"
            )
        _valid_surface_bulk = ("constant", "coare3", "large_yeager")
        if self.surface_bulk_scheme not in _valid_surface_bulk:
            errors.append(
                f"surface_bulk_scheme must be one of {_valid_surface_bulk}, "
                f"got {self.surface_bulk_scheme!r}"
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
        # Optional cloud-tuning override bounds (mirror CloudConfig.__param_spec__
        # so an out-of-range knob fails early, not deep in the cloud diagnosis).
        for _f, _lo, _hi in (
            ("cloud_rh_crit", 0.5, 0.99),
            ("cloud_q_c_diagnostic", 5.0e-5, 1.0e-3),
            ("cloud_conv_cloud_max", 0.1, 1.0),
        ):
            _v = getattr(self, _f)
            if _v is not None and not (_lo <= _v <= _hi):
                errors.append(
                    f"{_f}={_v!r} out of range [{_lo}, {_hi}]"
                )
        _valid_gwd = (
            "rayleigh", "lindzen", "mcfarlane", "hines",
            "prognostic_spectral", "e3sm_cam", "ml_emulator", "none",
        )
        if self.gravity_wave_drag not in _valid_gwd:
            errors.append(
                f"gravity_wave_drag must be one of {_valid_gwd}, "
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
            microphysics=amip_cfg.microphysics,
            convection=getattr(amip_cfg, 'convection', 'sbm'),
            turbulence=getattr(amip_cfg, 'turbulence', 'none'),
            gravity_wave_drag=getattr(amip_cfg, 'gravity_wave_drag', 'none'),
            fix_moisture=getattr(amip_cfg, 'fix_moisture', False),
            energy_consistent_moisture_clip=getattr(
                amip_cfg, 'energy_consistent_moisture_clip', False),
            topography=amip_cfg.topography,
            topo_smoothing=amip_cfg.topo_smoothing,
            topo_edge_blend=amip_cfg.topo_edge_blend,
            land_mask_path=getattr(amip_cfg, 'land_mask_path', ''),
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
            sbm_tau_c=amip_cfg.sbm_tau_c,
            sbm_RH_ref=amip_cfg.sbm_RH_ref,
            sbm_cape_threshold=getattr(amip_cfg, 'sbm_cape_threshold', 70.0),
            sigma_b=amip_cfg.sigma_b,
            k_BL_max_per_day=amip_cfg.k_BL_max_per_day,
            k_free_per_day=amip_cfg.k_free_per_day,
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
        return AMIPExperimentConfig(
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
            sbm_tau_c=self.sbm_tau_c,
            sbm_RH_ref=self.sbm_RH_ref,
            sbm_cape_threshold=self.sbm_cape_threshold,
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

    Sub-configs (grid, dycore, output) are inlined as nested dicts.
    This is the canonical serialization format.
    """
    d = config._asdict()
    for key in _SUB_CONFIGS:
        sub = d[key]
        if hasattr(sub, '_asdict'):
            d[key] = sub._asdict()
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
            known_sub = set(cls._fields)
            filtered = {k: v for k, v in d[key].items() if k in known_sub}
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
