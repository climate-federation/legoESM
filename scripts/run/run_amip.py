#!/usr/bin/env python
"""AMIP simulation via the canonical ``ModelDriver`` path.

Usage:
    JAX_ENABLE_X64=1 python scripts/run_amip.py \
        --dataset analytical --days 30 --resolution 16 --dt 600

    JAX_ENABLE_X64=1 python scripts/run_amip.py \
        --dataset cobe --forcing-path /path/to/MODEL.SST.COBE-SST2.nc \
        --radiation rrtmgp --convection sbm --turbulence louis \
        --days 30 --resolution 16 --dt 600
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Enable line-buffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

# Early JAX distributed init — must happen before any legoESM/JAX import that
# triggers XLA backend discovery (jax.numpy import in core/precision.py).
# Route-B multicontroller (--multicontroller) initializes jax.distributed in
# main() via init_multicontroller_distributed's explicit --coordinator path (a
# direct jax.distributed.initialize, no mpi4py); skip this import-time MPI
# auto-detect for it — it would try to load libmpi before argv is parsed and
# hard-crash on a node without a loadable MPI library (verified).
from legoesm.parallel.early_init import maybe_init_jax_distributed

if "--multicontroller" not in sys.argv:
    maybe_init_jax_distributed()

from legoesm.driver.config import (
    VALID_RADIATION,
    VALID_TURBULENCE,
    DycoreConfig,
    EvaluationConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
    parse_gwd_spec,
)
from legoesm.driver.run_status import status_to_exit_code

from legoesm import constants

_DYCORE_DEFAULTS = DycoreConfig()
_OUTPUT_DEFAULTS = OutputConfig()
_EXPERIMENT_DEFAULTS = ExperimentConfig()
_EVALUATION_DEFAULTS = EvaluationConfig()


def _print_forcing_activity(args) -> None:
    """Print a forcing-channel activity summary (call on rank-0 only).

    Mirrors the table in ``run_amip_cmip6_deck.py`` for the direct AMIP
    path.  GHG/ozone/aerosol/volcanic are gated on rrtmg/rrtmgp radiation;
    SST/SIC is always active; solar file threading is active when
    ``--solar-source`` is file-based.
    """
    rad_active = getattr(args, "radiation", "gray") in ("rrtmg", "rrtmgp")
    aerosol_active = getattr(args, "aerosol_forcing", "off") == "external"
    volcanic_active = (
        aerosol_active
        and bool(getattr(args, "volcanic_aerosol_file", ""))
        and getattr(args, "volcanic_aerosol_scale", 0.0) > 0.0
    )
    solar_file_active = getattr(args, "solar_source", "constant") in (
        "file", "spectral_file"
    )

    def _flag(active: bool) -> str:
        return "ACTIVE" if active else "inert  (gray radiation)"

    print("[run_amip] Forcing-channel activity for this run:")
    print("  SST/SIC                              ACTIVE        (radiation-independent)")
    if solar_file_active:
        print("  Solar TSI                            ACTIVE        (time-varying from file)")
    else:
        print("  Solar TSI                            constant S_0  (--solar-source constant)")
    print(f"  Greenhouse gases (transient annual)  "
          f"{_flag(rad_active and getattr(args, 'ghg_forcing', 'constant') == 'external')}")
    print(f"  Ozone (cyclic clim or interannual)   "
          f"{_flag(rad_active and getattr(args, 'ozone_forcing', 'inline') == 'external')}")
    if aerosol_active:
        print(f"  Tropospheric aerosol (Kinne)         {_flag(rad_active)}")
    if volcanic_active:
        print(f"  Volcanic stratospheric AOD           {_flag(rad_active)}")
    if not rad_active:
        print(
            "[run_amip] NOTE: --radiation gray disables GHG/ozone/aerosol/"
            "volcanic. Use --radiation rrtmg for production AMIP."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AMIP simulation with prescribed SST/SIC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Run config file (shared --config mechanism; keys set argument DEFAULTS so
    # any explicit CLI flag still overrides the file). The authoritative AMIP
    # production config lives at config/amip/amip_production.yaml.
    parser.add_argument("--config", default=None,
                        help="YAML run-config file (e.g. "
                             "config/amip/amip_production.yaml): its keys set "
                             "argument DEFAULTS, so any explicit CLI flag still "
                             "overrides it. Keys are run_amip argument dests; an "
                             "unknown key is a hard error (no silent typo'd "
                             "override).")
    parser.add_argument("--params", default=None,
                        help="YAML calibration file of tuned parameters keyed by "
                             "param_collector qualified name 'scheme_key.field' "
                             "(e.g. atm.clouds.CloudConfig.q_c_diagnostic); "
                             "validated against __param_spec__ bounds and applied "
                             "to the flattened atmosphere ExperimentConfig scalar "
                             "fields. Applied after --config/CLI. (issue #691)")

    # Forcing
    parser.add_argument("--dataset", type=str, default="analytical",
                        choices=["cobe", "hadisst", "custom", "analytical"])
    parser.add_argument("--forcing-path", type=str, default=None)
    parser.add_argument("--sic-path", type=str, default=None,
                        help="Separate SIC file (for ICON split SST/SIC files)")
    parser.add_argument("--sst-var", type=str, default=None)
    parser.add_argument("--sic-var", type=str, default=None)
    parser.add_argument("--time-var", type=str, default=None)
    parser.add_argument("--lat-var", type=str, default=None)
    parser.add_argument("--lon-var", type=str, default=None)
    parser.add_argument("--sst-offset", type=float, default=None)
    parser.add_argument("--sic-scale", type=float, default=None)

    # Grid
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--nlev", type=int, default=40)
    parser.add_argument("--vertical-coord", type=str, default="hybrid",
                        choices=["sigma", "hybrid"])
    parser.add_argument("--p-top", type=float, default=None)
    parser.add_argument("--stretching", type=float, default=None)
    # ``mpas`` is the canonical name for the SCVT Voronoi mesh + TRiSK
    # discretization (Ringler 2010 / Thuburn 2009), matching the ocean
    # side which has always used this name.  Legacy aliases
    # ``voronoi`` / ``icosahedral`` / ``mpas_voronoi`` are accepted and
    # normalised by ``legoesm.driver.config.normalize_grid_type``
    # before reaching any internal dispatch.
    # default=None is a sentinel meaning "not explicitly set" so that
    # _postprocess_args can distinguish a real user choice from the production
    # default (cubed_sphere) when checking --truncation/--discretization
    # conflicts; it resolves the sentinel to "cubed_sphere" afterwards.
    parser.add_argument("--grid-type", type=str, default=None,
                        choices=["cubed_sphere", "gaussian", "latlon",
                                 "mpas",
                                 "voronoi", "icosahedral", "mpas_voronoi"])
    parser.add_argument("--use-duogrid", action="store_true", default=False,
                        help="Enable FV3 Duo-Grid halo exchange (required for MPI multi-node)")
    # The canonical names in `supported_matrix.py` are:
    #   - centered       (cubed_sphere, latlon)
    #   - finite_volume  (cubed_sphere, latlon)
    #   - cdgrid         (cubed_sphere only)
    #   - latlon_cgrid   (latlon only)  — was 'cgrid' below; alias kept
    #   - mpas           (voronoi only)
    #   - spectral       (gaussian only)
    # Both 'cgrid' (legacy) and 'latlon_cgrid' (canonical) are accepted;
    # the postprocessor canonicalises 'cgrid' → 'latlon_cgrid' so the
    # downstream factory finds a matching ``(model_type, discretization,
    # grid_type)`` triple.
    # default=None sentinel: see --grid-type; resolves to "centered".
    parser.add_argument("--discretization", type=str, default=None,
                        choices=["centered", "finite_volume", "cgrid",
                                  "latlon_cgrid", "cdgrid", "mpas", "spectral"])
    parser.add_argument("--truncation", type=int, default=None,
                        help="Spectral truncation (T21, T42, etc.). Sets grid_type=gaussian.")

    # Integration
    parser.add_argument("--start-day", type=float, default=0.0)
    parser.add_argument("--days", type=int, default=200)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument(
        "--dt-auto", action="store_true", default=False,
        help="Override --dt with the empirically-validated cross-grid "
             "stability-ladder value for (grid_type, resolution) from "
             "legoesm.driver.rce_dt.auto_dt_rce (e.g. C36->150 s, "
             "latlon72->75 s, voronoi->300 s).  The ladder-safe choice "
             "for long production runs; raises if the resolution is "
             "past the last measured ladder point.")
    # Issue #273 Phase 3: implicit gravity-wave damping (semi-implicit
    # Helmholtz solve via CG).  Off by default to preserve bit-exact
    # behaviour with the legacy explicit-diffusion path.  Set
    # ``--implicit-grav-wave-use-pcg --implicit-grav-wave-damping
    # 1e8`` (typical α ~ 1e7–1e8 m²/s) to remove the explicit-CFL
    # ceiling and enable larger ``--dt``.
    # Stage 3-E: Fourier polar filter for lat-lon C-grid.  Lifts the
    # pole-cell CFL constraint by truncating high-wavenumber Fourier
    # modes near the poles, so ``--dt`` can be set by the equatorial
    # CFL.  Essential for 1° AMIP runs spanning >10 yr.
    parser.add_argument(
        "--use-polar-filter", action=argparse.BooleanOptionalAction,
        help="Enable Fourier polar filter for lat-lon C-grid (lifts "
             "pole-cell CFL → enables larger --dt at high resolution).",
    )
    parser.add_argument(
        "--polar-filter-cutoff-deg", type=float, default=60.0,
        help="Latitude (degrees) poleward of which the polar filter "
             "is applied (default 60.0).",
    )
    parser.add_argument(
        "--polar-filter-max-wave-speed", type=float, default=300.0,
        help="Max wave speed [m/s] used to size the polar filter "
             "CFL mask (default 300.0 = external gravity wave).",
    )
    # #836: hydrostatic lat-lon C-grid top sponge (Rayleigh damping increasing
    # toward the model lid; absorbs upward gravity-wave energy).  Default OFF.
    parser.add_argument(
        "--sponge-coeff", type=float, default=_DYCORE_DEFAULTS.sponge_coeff,
        help="Rayleigh top-sponge damping SCALE [1/s] for the hydrostatic "
             "lat-lon C-grid (0 = OFF, default; e.g. 1.157e-5 = 1/day). Exact "
             "lid value for --sponge-shape sin2; sam_rational peaks at "
             "sponge_coeff*100/101. Absorbs gravity-wave energy reflecting "
             "off the rigid model lid (#836).",
    )
    parser.add_argument(
        "--sponge-width-m", type=float, default=_DYCORE_DEFAULTS.sponge_width_m,
        help="Top-sponge layer depth below the model lid [m] (default "
             f"{_DYCORE_DEFAULTS.sponge_width_m}).",
    )
    parser.add_argument(
        "--sponge-shape", type=str, default=_DYCORE_DEFAULTS.sponge_shape,
        choices=["sin2", "sam_rational"],
        help="Top-sponge ramp shape (default 'sin2').",
    )
    parser.add_argument(
        "--sponge-scale-height-m", type=float,
        default=_DYCORE_DEFAULTS.sponge_scale_height_m,
        help="Log-pressure scale height [m] mapping sigma->z for the top "
             f"sponge (default {_DYCORE_DEFAULTS.sponge_scale_height_m}).",
    )
    # Task #25: JIT compile bloat at production scale.  The inline
    # SSP-RK3 calls tendency_fn 3× sequentially → XLA inlines three
    # copies of the entire tendency pipeline.  Folding the 3 stages
    # into a single ``lax.scan`` body cuts the jaxpr ~2× and the
    # compile time 1.3–1.9× (measured on lat-lon C-grid + 3 tracers
    # + polar filter, profile job 8070275).  Same RK3 coefficients,
    # bit-equivalent output (pinned by
    # tests/timestepping/test_ssp_rk3_scan_bit_equivalence.py).
    parser.add_argument(
        "--time-integrator", type=str, default="auto",
        choices=["auto", "ssp_rk3", "ssp_rk3_scan", "ssp_rk34",
                 "ssp_rk54", "ssp_rk54_scan", "rk4"],
        help="Time integrator.  'auto' (default) selects each dycore's "
             "own stable default: ssp_rk3 on cube/lat-lon (IEEE-"
             "identical to existing runs) and ssp_rk54_scan on MPAS "
             "(the biharmonic hyperdiffusion eigenvalues at production "
             "dt fall outside ssp_rk3's stability region — the former "
             "'hidden CFL' blow-up).  An explicit name is forwarded "
             "verbatim to every dycore, including ssp_rk3 on MPAS for "
             "deliberate integrator-sensitivity runs.  ssp_rk3_scan is "
             "the JIT-compile-time optimised variant for production "
             "lat-lon C-grid AMIP.",
    )
    parser.add_argument(
        "--implicit-grav-wave-use-pcg", action="store_true",
        default=False,
        help=(
            "Issue #273 Phase 3: switch the post-RK3 gravity-wave "
            "damping from explicit forward-Euler to an implicit "
            "Helmholtz solve via jax.scipy.sparse.linalg.cg.  Removes "
            "the CFL ceiling on ``--implicit-grav-wave-damping`` and "
            "enables larger ``--dt``."
        ),
    )
    parser.add_argument(
        "--implicit-grav-wave-damping", type=float, default=0.0,
        help=(
            "Gravity-wave damping coefficient α [m²/s].  ``0`` (default) "
            "skips the post-RK3 ``p_s`` damping entirely.  Typical "
            "production: α ~ 1e7–1e8.  At α dt / dx² > 0.5 you MUST "
            "also pass ``--implicit-grav-wave-use-pcg`` or the "
            "explicit path will blow up."
        ),
    )
    parser.add_argument("--diag-days", type=int, default=5)
    parser.add_argument("--hyperdiff-scale", type=float,
                        default=_DYCORE_DEFAULTS.hyperdiff_scale,
                        help="Dycore hyperdiffusion multiplier")
    parser.add_argument("--mpas-nu-vert4-t", type=float,
                        default=_DYCORE_DEFAULTS.mpas_nu_vert4_T,
                        help="MPAS vertical biharmonic hyperdiffusion of T "
                             "[1/s] — #930 2Δσ vertical-checkerboard cure "
                             "(0 disables)")
    parser.add_argument("--div-damp-scale", type=float,
                        default=_DYCORE_DEFAULTS.div_damp_scale,
                        help="Dycore divergence-damping multiplier")
    parser.add_argument("--conservation-fixer",
                        action=argparse.BooleanOptionalAction,
                        default=_DYCORE_DEFAULTS.conservation_fixer,
                        help="Enable/disable the dycore conservation fixer")
    parser.add_argument("--fix-mass",
                        action=argparse.BooleanOptionalAction,
                        default=_DYCORE_DEFAULTS.fix_mass,
                        help="Enable/disable global mass correction")
    parser.add_argument("--moisture-flux-form",
                        action=argparse.BooleanOptionalAction,
                        default=_DYCORE_DEFAULTS.moisture_flux_form,
                        help="#771 (EXPERIMENTAL, cubed_sphere cdgrid + "
                             "--moisture-advection only): transport moisture "
                             "with the mass-conserving flux-form post-RK3 "
                             "substep instead of the advective -(u.grad q). "
                             "Default off = advective (bit-exact).")

    # Output
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--checkpoint-days", type=int, default=0)
    parser.add_argument("--aimip-classical-checkpoint", type=str, default=None,
                        help="Path to an AIMIP-classical trained params .eqx "
                             "(e.g. results/aimip_001/classical/epoch_0019.eqx). "
                             "Forces the classical scheme set (tiedtke / louis / "
                             "mcfarlane / xu_randall) and seeds them with the "
                             "trained best-fit values as INITIAL parameters.")
    parser.add_argument("--restart-from", type=str, default=None)
    parser.add_argument("--restart-start-day", type=float, default=None,
                        help="Override start_day after loading checkpoint. "
                             "Pass 0.0 at year boundaries to reset the day counter.")
    parser.add_argument("--checkpoint-format", type=str,
                        default=_OUTPUT_DEFAULTS.checkpoint_format,
                        choices=["npz", "zarr"],
                        help="Restart checkpoint serialization format")
    parser.add_argument("--max-wallclock-seconds", type=float,
                        default=_OUTPUT_DEFAULTS.max_wallclock_seconds,
                        help="Wallclock budget [s] for clean checkpoint+exit")
    parser.add_argument("--restart-buffer-seconds", type=float,
                        default=_OUTPUT_DEFAULTS.restart_buffer_seconds,
                        help="Wallclock buffer [s] reserved for restart writes")

    # Initial atmospheric state
    parser.add_argument("--t-init", type=float, default=None,
                        help="Initial isothermal temperature [K] (default 300)")
    parser.add_argument("--rh-init", type=float, default=None,
                        help="Initial relative humidity (default 0.7); lower for drier IC")
    parser.add_argument("--ic", type=str, default="default",
                        choices=["default", "standard", "era5"],
                        help="Initial condition source: 'default' uses a uniform "
                             "T_init rest state; 'standard' uses a realistic "
                             "constant-lapse-rate atmosphere with an equator-pole "
                             "surface-temperature gradient (Earth-like CWV); "
                             "'era5' loads reanalysis from --ic-path")
    parser.add_argument("--ic-path", type=str, default="",
                        help="Path to ERA5 Zarr store for --ic era5")

    # Radiation
    # Derived from the canonical tuple so this list cannot drift from
    # validate_strict.  "none" is REQUIRED here, not decorative: this driver
    # has a --held-suarez-forcing lane, HS forcing is ADDITIVE to the physics
    # pipeline rather than a replacement, and a genuine dry HS run therefore
    # needs radiation OFF.  Omitting it made the dry-core lane unreachable from
    # this driver (scheme-reachability audit).
    parser.add_argument("--radiation", type=str, default="gray",
                        choices=list(VALID_RADIATION),
                        help="Radiation scheme (default: gray). 'none' is for "
                             "the dry --held-suarez-forcing lane.")
    # Default is ``None`` so ``_postprocess_args`` can tell an explicit
    # ``--rad-update-steps 1`` from "the user did not pass this flag".
    # ``--production-profile`` only auto-sets the production cadence
    # when the user did not provide a value.  Resolved to ``1`` after
    # production-profile processing.
    parser.add_argument("--rad-update-steps", type=int, default=None)
    parser.add_argument("--unfused-radiation", action="store_true", default=False,
                        help="Run radiation outside the compiled segment scan")
    parser.add_argument("--per-step-rollout", action="store_true", default=False,
                        help="Use the per-step Python-loop rollout "
                             "(driver.run(compiled=False)) instead of the "
                             "compiled lax.scan segments. Slower, but threads the "
                             "CLUBB cloud-fraction->radiation carry "
                             "(--use-clubb-cloud-fraction), which the compiled "
                             "SegmentCarry path does not yet carry.")
    parser.add_argument("--rrtmgp-gpoint-batch-size", type=int,
                        default=_EXPERIMENT_DEFAULTS.rrtmgp_gpoint_batch_size,
                        help="RRTMGP g-point batch size (0 = auto/checkpointed)")
    parser.add_argument("--no-rrtmgp-gpoint-checkpoint",
                        dest="rrtmgp_gpoint_checkpoint",
                        action="store_false",
                        default=_EXPERIMENT_DEFAULTS.rrtmgp_gpoint_checkpoint,
                        help="Disable per-g-point jax.checkpoint(prevent_cse=True) "
                             "in the RRTMGP two-stream scan. False = plain lax.scan: "
                             "smaller compiled footprint / faster cold compile for "
                             "FORWARD/inference runs (relieves the MPAS/L5 XLA compile "
                             "wall). Keep True for reverse-mode AD / training.")
    parser.add_argument("--radiation-column-chunk", type=int,
                        default=_EXPERIMENT_DEFAULTS.rrtmgp_column_chunk_size,
                        help="RRTMGP column-chunk block size (0 = off). >0 maps "
                             "the rrtmgp solve over fixed-size column blocks so the "
                             "per-block XLA graph compiles ONCE at this size — caps "
                             "the super-linear rrtmgp compile time so higher "
                             "resolutions (C24/C48 L20) compile instead of stalling. "
                             "Numerically exact (columns are independent); must "
                             "divide the column count.")
    # Issue #273 GPU tuning: RRTMGP column-recurrence kernel choice.
    # ``--rrtmgp-use-scan`` forces ``jax.lax.scan`` (smaller graph,
    # ~5-10× cheaper to JIT — material against the 2600s cold compile
    # called out in the issue); ``--rrtmgp-no-scan`` forces the
    # Python for-loop unroll.  Neither flag → auto-pick (scan on
    # GPU/TPU, unroll on CPU/Metal), which is the new production
    # default.
    _rrtmg_scan = parser.add_mutually_exclusive_group()
    _rrtmg_scan.add_argument(
        "--rrtmgp-use-scan", dest="rrtmgp_use_scan",
        action="store_const", const=True, default=None,
        help=(
            "Force RRTMGP column recurrence to use jax.lax.scan.  "
            "Default (no flag) auto-picks scan on GPU/TPU."
        ),
    )
    _rrtmg_scan.add_argument(
        "--rrtmgp-no-scan", dest="rrtmgp_use_scan",
        action="store_const", const=False,
        help=(
            "Force RRTMGP column recurrence to use a Python for-loop "
            "(legacy default).  Useful for CPU benchmarking."
        ),
    )
    parser.add_argument("--diurnal-cycle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--orbital-insolation", action=argparse.BooleanOptionalAction, default=False,
        dest="orbital_insolation",
        help="Use realistic (Berger 1978) orbital insolation for AMIP-II: "
             "present-day orbital declination + Earth-Sun distance factor "
             "(a/r)^2 eccentricity asymmetry (~+/-3.4%%). Default off = "
             "circular orbit (idealized).",
    )
    parser.add_argument("--dynamic-albedo", action=argparse.BooleanOptionalAction, default=False,
                        help="Zenith-angle-dependent ocean albedo "
                             "(Briegleb 1992) instead of the constant "
                             "ocean albedo; sea-ice/land blends unchanged.")
    parser.add_argument("--co2-ppmv", type=float, default=415.0)
    parser.add_argument("--ch4-ppbv", type=float, default=1900.0)
    parser.add_argument("--n2o-ppbv", type=float, default=332.0)
    parser.add_argument("--solar-s0", type=float, default=constants.S_0,
                        help="Total solar irradiance [W/m^2]")
    parser.add_argument("--tau-equator", type=float,
                        default=_EXPERIMENT_DEFAULTS.tau_equator,
                        help="Gray-radiation equatorial optical depth")
    parser.add_argument("--tau-pole", type=float,
                        default=_EXPERIMENT_DEFAULTS.tau_pole,
                        help="Gray-radiation polar optical depth")
    parser.add_argument("--ozone-source", type=str, default="standard",
                        choices=["standard", "analytical", "none"])
    parser.add_argument("--ozone-forcing", type=str, default="inline",
                        choices=["inline", "external", "off"])
    parser.add_argument(
        "--ozone-file", type=str, default="",
        help=(
            "Path to an ozone NetCDF/Zarr file (CMIP6 input4MIPs "
            "compatible).  Auto-detects variable name among "
            "'ozone'/'vmro3'/'o3'/'O3'/'tro3'.  Files with 12 months of "
            "data are interpreted as a monthly climatology and looped "
            "every year; files with >12 months (e.g. the CMIP6 "
            "1850–2014 vmro3 file with 1980 months and "
            "'months since 1850-01-01' CF units) are interpreted as an "
            "interannually varying dataset and dispatched through "
            "OzoneConfig.start_year to preserve the real time evolution."
        ))
    parser.add_argument("--ghg-forcing", type=str, default="constant",
                        choices=["constant", "external"])
    parser.add_argument("--ghg-file", type=str, default="",
                        help="Path to a time-varying GHG forcing file")

    # Solar
    parser.add_argument("--solar-source", type=str, default="constant",
                        choices=["constant", "file", "spectral_file"],
                        help=(
                            "'constant' uses --co2/ch4/n2o/solar-S_0 "
                            "defaults; 'file' reads a 1-D TSI time series; "
                            "'spectral_file' additionally reads a 2-D "
                            "(time, band|gpt) spectrum."
                        ))
    parser.add_argument("--solar-file", type=str, default="")
    parser.add_argument(
        "--solar-tsi-var", type=str, default="tsi",
        help=(
            "Variable name for broadband TSI [W/m^2] in the solar forcing "
            "file.  Case-insensitive: the canonical MPI-M CMIP6 spectral-"
            "solar file stores 'TSI', the in-tree default is 'tsi' — "
            "either works."
        ))
    parser.add_argument(
        "--solar-spectral-var", type=str,
        default="solar_fraction_by_gpt",
        help=(
            "Variable name for the normalised spectrum in "
            "--solar-source=spectral_file.  Case-insensitive lookup.  "
            "If the loaded axis has <50 entries (e.g. CMIP6 'SSI_frac' "
            "with 14 solar bands) it is expanded to per-g-point weights "
            "(112 g-points on the default RRTMG-SW table) so the "
            "radiation solver receives one weight per g-point."
        ))
    parser.add_argument(
        "--solar-spectral-band-order", type=str, default="auto",
        choices=["auto", "as_is", "rrtmg_sw"],
        help=(
            "Band ordering of a per-band (14-band) --solar-spectral-var "
            "input (issue #322).  'auto' (default) rotates to RRTMGP "
            "order only when the input carries the MPI-M CMIP6 signature "
            "(SSI_frac variable / swflux_14band filename) and leaves "
            "generic files untouched.  'rrtmg_sw' always rotates (file "
            "in RRTMG-SW / CMIP order, 820-2680 cm^-1 band last); "
            "'as_is' never rotates (file already in RRTMGP order)."
        ))

    # Aerosol
    parser.add_argument("--aerosol-forcing", type=str, default="off",
                        choices=["off", "external"])
    parser.add_argument("--aerosol-file", type=str, default="")
    parser.add_argument("--aerosol-reference-aod", type=float, default=0.03)
    parser.add_argument(
        "--volcanic-aerosol-file", type=str, default="",
        help=(
            "Path to a volcanic stratospheric aerosol file.  Auto-"
            "dispatched by file schema: the legacy 'aod(time, lat)' "
            "convention is read directly, while the CMIP6 / MPI-M files "
            "'bc_aeropt_cmip6_volc_lw_b16_sw_b14_<year>.nc' storing "
            "per-band 'ext_sun(solar_bands, lat, altitude, month)' in "
            "[1/km] are integrated over altitude (∫ ext·dz) and averaged "
            "over SW bands to produce a representative single-band AOD "
            "(matching the Kinne multi-band aerosol convention)."
        ))
    parser.add_argument("--volcanic-aerosol-scale", type=float, default=1.0)
    parser.add_argument("--volcanic-aerosol-lw", action="store_true",
                        default=False,
                        help="Load and apply volcanic longwave aerosol optical depth")

    # Subgrid physics
    parser.add_argument("--convection", type=str, default="tiedtke",
                        choices=[
                            "none", "sbm", "dca", "kuo", "mass_flux", "edmf",
                            "zhang_mcfarlane", "kain_fritsch", "emanuel",
                            "tiedtke", "bechtold",
                        ])
    # Full-physics policy: an AMIP run is a real atmosphere, so every
    # parameterization defaults to an ACTIVE scheme — never "none".  A slot is
    # disabled only for an idealized / dry-dynamics run, which must opt in via
    # --allow-disabled-physics (or --held-suarez-forcing / --enable-latlon-spmd).
    # Enforced by _require_full_physics_for_amip; see the directive in CLAUDE.
    # mynn25 was missing while run_coupled offers it, and it resolves through
    # the shared turbulence factory (integration.get_turbulence_fn) -- drift.
    # Derived from the canonical tuple: this hand-copied list is exactly what
    # drifted (it silently omitted mynn25 while run_coupled's copy omitted
    # clubb_lite/ysu -- same ModelDriver, same factory, two different answers to
    # "what can I run?").
    parser.add_argument("--turbulence", type=str, default="louis",
                        choices=list(VALID_TURBULENCE))
    parser.add_argument("--cloudtop-entrainment-efficiency",
                        dest="louis_cloudtop_entrainment_efficiency", type=float,
                        default=0.0,
                        help="Marine-Sc cloud-top entrainment efficiency A in "
                             "[0,1] for the Louis PBL (K_ent = A*W_REF*dz*gates, "
                             "W_REF=0.02 m/s): vents trapped BL-top moisture into "
                             "the dry free troposphere to thin excess stratocumulus "
                             "liquid cloud (the AMIP albedo bias) WITHOUT a "
                             "surface-evaporation trade.  0 = off (default); "
                             "warm-start/ramp only (cold-start caveat).")
    # Shared validator: composites keep working and a typo is now rejected at
    # the CLI (this flag previously had `type=str` with no validation at all).
    parser.add_argument("--gravity-wave-drag", type=parse_gwd_spec,
                        default="mcfarlane",
                        help="GWD scheme: none, rayleigh, lindzen, mcfarlane, "
                             "hines, prognostic_spectral, ml_emulator, or a "
                             "'+'-joined composite whose source tendencies are "
                             "summed. Composable parts: rayleigh, lindzen, "
                             "mcfarlane, hines, and (as the single stateful "
                             "member) prognostic_spectral — e.g. "
                             "'mcfarlane+prognostic_spectral' to run orographic "
                             "+ non-orographic GWD together (issue #834), or "
                             "'hines+mcfarlane'. Validated in ExperimentConfig.")
    # Tuned air-sea + cloud knobs (the CMIP-realism calibration) — mirror
    # run_coupled so AMIP can run with the SAME tuned slab parameters. Defaults
    # (constant / 0 / None / off) keep the prior AMIP behaviour byte-identical.
    parser.add_argument("--surface-bulk-scheme", type=str, default="constant",
                        choices=["constant", "coare3", "large_yeager"],
                        help="Surface-layer bulk-flux scheme (coare3 = COARE 3.0 "
                             "MOST with convective gustiness; the tuned slab value). "
                             "Matches ExperimentConfig.validate_strict — 'most' is "
                             "not an accepted AMIP surface scheme (coare3 is the "
                             "MOST-with-gustiness variant).")
    parser.add_argument("--gustiness-zi", dest="surface_gustiness_zi", type=float,
                        default=None,
                        help="COARE convective-gustiness BL depth z_i [m]. "
                             "Unset = scheme-native (coare3: 600 m per "
                             "AeroBulk/Fairall 2003, others: off); 0 = force "
                             "off; tuned slab value 300.")
    parser.add_argument("--bulk-thermo-convention", dest="bulk_thermo_convention",
                        type=str, default="legoesm",
                        choices=["legoesm", "aerobulk"],
                        help="Thermodynamic constants set for the MOST bulk "
                             "fluxes (coare3/large_yeager): 'legoesm' "
                             "(default) = constant L_v / dry c_pd; 'aerobulk' "
                             "= NEMO/AeroBulk/COARE parity (SST-dependent "
                             "L_vap, moist cp_air).")
    parser.add_argument("--q-c-diagnostic", dest="cloud_q_c_diagnostic", type=float,
                        default=None,
                        help="In-cloud diagnostic condensate fed to radiation "
                             "[kg/kg] (None=CloudConfig default; tuned slab 3e-4).")
    parser.add_argument("--rh-crit", dest="cloud_rh_crit", type=float, default=None,
                        help="Critical RH for cloud onset (None=scheme default).")
    parser.add_argument("--cloud-inhomogeneity-factor",
                        dest="cloud_inhomogeneity_factor", type=float, default=None,
                        help="Cahalan (1994) horizontal-inhomogeneity factor chi "
                             "[0.3,1.0] scaling the radiative cloud water path "
                             "(plane-parallel albedo bias). LOWER => thinner "
                             "optics => lower albedo. None=CloudConfig default 1.0 "
                             "(homogeneous). ~0.7 is the observed correction. "
                             "Only used when --cloud-optics-inhomogeneity=constant.")
    parser.add_argument("--cloud-optics-inhomogeneity",
                        dest="cloud_optics_inhomogeneity",
                        choices=["constant", "two_region"], default="constant",
                        help="Sub-grid cloud-optics inhomogeneity scheme: "
                             "'constant' (Cahalan scalar chi, legacy default) or "
                             "'two_region' (TAU-DEPENDENT Shonk-Hogan 2008 optic "
                             "that breaks the plane-parallel tau-saturation a "
                             "scalar cannot -- a thick cloud is reduced MORE than "
                             "a thin one).")
    parser.add_argument("--cloud-fsd", dest="cloud_fsd", type=float, default=None,
                        help="Fractional std-dev of in-cloud water for the "
                             "two_region optic [0,1] (Shonk-Hogan ~0.75; HIGHER "
                             "=> thinner leaking sub-column => lower albedo). "
                             "None=CloudConfig default 0.75.")
    parser.add_argument("--use-clubb-cloud-fraction",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Route diagnostic CLUBB's sub-grid PDF cloud "
                             "fraction into the cloud optics instead of the RH "
                             "grid-scale one (marine-Sc over-bright albedo lever; "
                             "a moist closure is less overcast over a saturated "
                             "marine BL => lower LWP floor => lower albedo). "
                             "Requires --turbulence clubb (diagnostic). Default "
                             "off = RH grid-scale cloud fraction (byte-identical).")
    parser.add_argument("--cloud-p-xr", dest="cloud_p_xr", type=float, default=None,
                        help="Xu-Randall cloud-fraction RH exponent p_xr (None="
                             "default 0.25; bounds 0.05..1.0). HIGHER => cloud "
                             "fraction less saturating at moderate RH (flattens "
                             "the moisture-driven overcast runaway).")
    parser.add_argument("--cloud-alpha-xr", dest="cloud_alpha_xr", type=float,
                        default=None,
                        help="Xu-Randall condensate sensitivity alpha_xr (None="
                             "default 100; bounds 10..1000). LOWER => cloud "
                             "fraction grows more slowly with condensate.")
    parser.add_argument("--diagnostic-condensate-scheme",
                        dest="cloud_diagnostic_condensate_scheme",
                        choices=["constant", "adiabatic"], default="constant",
                        help="Vertical structure of the stratiform in-cloud "
                             "condensate floor. 'constant' (default) = flat "
                             "q_c_diagnostic at every cloudy level (validated). "
                             "'adiabatic' = depth-scaled adiabatic LWC that dims "
                             "THIN warm marine stratocumulus (the source-side "
                             "marine-BL albedo fix) while deep clouds stay at "
                             "the cap.")
    parser.add_argument("--adiabatic-lwc-rate", dest="cloud_adiabatic_lwc_rate",
                        type=float, default=None,
                        help="In-cloud LWC growth per metre of cloudy depth "
                             "[kg/kg/m] for --diagnostic-condensate-scheme="
                             "adiabatic (None=CloudConfig default 1.5e-6 ~ "
                             "1.5 g/kg per km; bounds 5e-7..3e-6).")
    parser.add_argument("--convective-cloud", dest="convective_cloud",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Add the convective (thin-cirrus) cloud-fraction "
                             "source (the tuned slab value is ON). Use "
                             "--no-convective-cloud to DISABLE a config-file "
                             "default: recommended for prescribed-SST AMIP, where "
                             "conv-cloud is an inert SST-drift compensator that "
                             "only adds planetary albedo (amip_production.yaml A/B: "
                             "OFF 0.295 vs ON 0.370).")
    parser.add_argument("--held-suarez-forcing", action="store_true", default=False)
    parser.add_argument("--allow-disabled-physics", action="store_true",
                        default=False,
                        help="Permit a parameterization slot set to 'none' (an "
                             "idealized / dry-dynamics run).  Without this flag an "
                             "AMIP run requires every physics slot active — see "
                             "_require_full_physics_for_amip.")
    parser.add_argument("--sbm-tau-c", type=float, default=7200.0)
    parser.add_argument("--sbm-rh-ref", type=float, default=0.7)
    parser.add_argument("--sbm-cape-threshold", type=float, default=70.0)
    parser.add_argument("--bechtold-cape-threshold", type=float,
                        default=_EXPERIMENT_DEFAULTS.bechtold_cape_threshold,
                        dest="bechtold_cape_threshold",
                        help="Bechtold deep-convection CAPE trigger threshold "
                             "[J/kg]; lower it to trigger convection more readily "
                             "at coarse resolution (the AMIP precip-deficit lever). "
                             f"Default {_EXPERIMENT_DEFAULTS.bechtold_cape_threshold}.")
    parser.add_argument("--bechtold-conv-top-pa", type=float,
                        default=_EXPERIMENT_DEFAULTS.bechtold_conv_top_pa,
                        dest="bechtold_conv_top_pa",
                        help="Bechtold convective-top pressure [Pa]: gates the "
                             "(non-detraining) plume + compensating-subsidence "
                             "above this cutoff (stability). 15000 (150 hPa) is a "
                             "tighter-than-kernel cap; raise toward 10000 (100 hPa) "
                             "if deep tropical tops are clipped. "
                             f"Default {_EXPERIMENT_DEFAULTS.bechtold_conv_top_pa}.")
    parser.add_argument("--bechtold-downdraft-evap", type=float,
                        default=_EXPERIMENT_DEFAULTS.bechtold_downdraft_evap,
                        dest="bechtold_downdraft_evap",
                        help="Bechtold convective-downdraft evaporation efficiency "
                             "[0,0.5] (marine-evaporation / precip lever, #847): "
                             "higher => the downdraft re-evaporates more rain => "
                             "more sub-cloud COOLING => cold pools enhance "
                             "convective triggering => more precip => net column "
                             "drying => larger sea-air gradient => higher surface "
                             f"evaporation. Default {_EXPERIMENT_DEFAULTS.bechtold_downdraft_evap} "
                             "(weak); Tiedtke ~0.3.")
    parser.add_argument("--bechtold-downdraft-alpha", type=float,
                        default=_EXPERIMENT_DEFAULTS.bechtold_downdraft_alpha,
                        dest="bechtold_downdraft_alpha",
                        help="Bechtold downdraft mass-flux fraction [0,0.9]. "
                             f"Default {_EXPERIMENT_DEFAULTS.bechtold_downdraft_alpha}.")
    parser.add_argument("--bechtold-downdraft-rh-min", type=float,
                        default=_EXPERIMENT_DEFAULTS.bechtold_downdraft_rh_min,
                        dest="bechtold_downdraft_rh_min",
                        help="Bechtold downdraft column-RH suppression threshold "
                             f"[0,1]. Default {_EXPERIMENT_DEFAULTS.bechtold_downdraft_rh_min}.")
    parser.add_argument("--bechtold-downdraft-transport",
                        dest="bechtold_downdraft_transport",
                        action=argparse.BooleanOptionalAction,
                        default=_EXPERIMENT_DEFAULTS.bechtold_downdraft_transport,
                        help="Enable the Bechtold PENETRATIVE downdraft: advect "
                             "low-MSE (dry) mid-level air DOWN into the sub-cloud "
                             "layer (Tiedtke 1989), DRYING the marine BL => "
                             "stronger surface evaporation + less BL liquid cloud "
                             "(lower albedo). UNLIKE --bechtold-downdraft-evap "
                             "(rain re-evaporation, which MOISTENS), this is the "
                             "BL-ventilation lever. --no-bechtold-downdraft-transport "
                             "disables a config-file default. Default "
                             f"{_EXPERIMENT_DEFAULTS.bechtold_downdraft_transport}.")
    parser.add_argument("--bechtold-use-ifs-cape-closure",
                        dest="bechtold_use_ifs_cape_closure",
                        action=argparse.BooleanOptionalAction,
                        default=_EXPERIMENT_DEFAULTS.bechtold_use_ifs_cape_closure,
                        help="Enable the full IFS deep CAPE closure "
                             "ZMFUB1=ZCAPE*ZMFUB/(ZHEAT*ZXTAU) (openifs "
                             "cumastrn.F90:704-833) instead of the legacy "
                             "surrogate deep closure. "
                             "--no-bechtold-use-ifs-cape-closure disables a "
                             "config-file default. Default "
                             f"{_EXPERIMENT_DEFAULTS.bechtold_use_ifs_cape_closure}.")
    parser.add_argument("--bechtold-use-ifs-subcloud-evap",
                        dest="bechtold_use_ifs_subcloud_evap",
                        action=argparse.BooleanOptionalAction,
                        default=_EXPERIMENT_DEFAULTS.bechtold_use_ifs_subcloud_evap,
                        help="Enable the IFS Kessler sub-cloud evaporation of "
                             "convective rain (openifs cuflxn.F90:436-475: "
                             "RCPECONS rate, ZRHEBC RH break) instead of the "
                             "crude downdraft-efficiency re-evaporation. "
                             "--no-bechtold-use-ifs-subcloud-evap disables a "
                             "config-file default. Default "
                             f"{_EXPERIMENT_DEFAULTS.bechtold_use_ifs_subcloud_evap}.")
    parser.add_argument("--bechtold-use-ifs-inplume-precip",
                        dest="bechtold_use_ifs_inplume_precip",
                        action=argparse.BooleanOptionalAction,
                        default=_EXPERIMENT_DEFAULTS.bechtold_use_ifs_inplume_precip,
                        help="Enable the IFS in-updraft precipitation formation "
                             "(openifs cuascn.F90:718-773 analytic Sundqvist "
                             "conversion; bypasses the post-hoc precip split). "
                             "--no-bechtold-use-ifs-inplume-precip disables a "
                             "config-file default. Default "
                             f"{_EXPERIMENT_DEFAULTS.bechtold_use_ifs_inplume_precip}.")
    parser.add_argument("--bechtold-use-ifs-downdraft",
                        dest="bechtold_use_ifs_downdraft",
                        action=argparse.BooleanOptionalAction,
                        default=_EXPERIMENT_DEFAULTS.bechtold_use_ifs_downdraft,
                        help="Enable the IFS convective downdraft (openifs "
                             "cudlfsn+cuddrafn: LFS, saturated entraining "
                             "descent, rain debit, closure/CMT coupling). "
                             "--no-bechtold-use-ifs-downdraft disables a "
                             "config-file default. Default "
                             f"{_EXPERIMENT_DEFAULTS.bechtold_use_ifs_downdraft}.")
    parser.add_argument("--bechtold-use-ifs-snow-melt",
                        dest="bechtold_use_ifs_snow_melt",
                        action=argparse.BooleanOptionalAction,
                        default=_EXPERIMENT_DEFAULTS.bechtold_use_ifs_snow_melt,
                        help="Enable the IFS convective snow partition + melt "
                             "(openifs cuflxn.F90 FOEALFCU wet-bulb split, "
                             "RTAUMEL melt; FOLD variant). Default "
                             f"{_EXPERIMENT_DEFAULTS.bechtold_use_ifs_snow_melt}.")
    parser.add_argument("--bechtold-use-ifs-capdcycl",
                        dest="bechtold_use_ifs_capdcycl",
                        action=argparse.BooleanOptionalAction,
                        default=_EXPERIMENT_DEFAULTS.bechtold_use_ifs_capdcycl,
                        help="Enable the IFS RCAPDCYCL=2 diurnal-cycle CAPE "
                             "correction (openifs cumastrn.F90:780-833; land "
                             "deep convection peaks late afternoon). Default "
                             f"{_EXPERIMENT_DEFAULTS.bechtold_use_ifs_capdcycl}.")
    parser.add_argument("--bechtold-use-ifs-land-rhebc",
                        dest="bechtold_use_ifs_land_rhebc",
                        action=argparse.BooleanOptionalAction,
                        default=_EXPERIMENT_DEFAULTS.bechtold_use_ifs_land_rhebc,
                        help="Enable the IFS land RH break for sub-cloud rain "
                             "evaporation (cuflxn.F90 0.70/0.75 land vs "
                             "0.85/0.92 ocean). Default "
                             f"{_EXPERIMENT_DEFAULTS.bechtold_use_ifs_land_rhebc}.")
    parser.add_argument("--bechtold-use-ifs-shallow-closure",
                        dest="bechtold_use_ifs_shallow_closure",
                        action=argparse.BooleanOptionalAction,
                        default=_EXPERIMENT_DEFAULTS.bechtold_use_ifs_shallow_closure,
                        help="Enable the IFS shallow PBL-equilibrium closure "
                             "(openifs cumastrn.F90 ZDHPBL/ZDH; flux-form "
                             "supply from same-step bulk SHF+LHF + sub-cloud "
                             "radiative convergence). "
                             "--no-bechtold-use-ifs-shallow-closure disables "
                             "a config-file default. Default "
                             f"{_EXPERIMENT_DEFAULTS.bechtold_use_ifs_shallow_closure}.")
    parser.add_argument("--bechtold-dx-m", type=float,
                        default=_EXPERIMENT_DEFAULTS.bechtold_dx_m,
                        dest="bechtold_dx_m",
                        help="Grid spacing [m] for the IFS ZTAURES convective-"
                             "turnover resolution factor (cumastrn.F90:762-768;"
                             " ZDX=sqrt(cell area)). 0 disables (legacy "
                             "factor 1.0). Default "
                             f"{_EXPERIMENT_DEFAULTS.bechtold_dx_m}.")
    parser.add_argument("--bechtold-downdraft-entrain-rate", type=float,
                        default=_EXPERIMENT_DEFAULTS.bechtold_downdraft_entrain_rate,
                        dest="bechtold_downdraft_entrain_rate",
                        help="Penetrative-downdraft fractional entrainment rate "
                             "[1/m] (mixes it toward the environment as it sinks; "
                             "larger => arrives less dry => weaker BL drying). "
                             f"Default {_EXPERIMENT_DEFAULTS.bechtold_downdraft_entrain_rate}.")
    parser.add_argument("--bechtold-downdraft-detrain-scale", type=float,
                        default=_EXPERIMENT_DEFAULTS.bechtold_downdraft_detrain_scale_m,
                        dest="bechtold_downdraft_detrain_scale_m",
                        help="Near-surface height scale [m] over which the "
                             "penetrative-downdraft mass flux tapers to zero (the "
                             "drying-deposit depth). Default "
                             f"{_EXPERIMENT_DEFAULTS.bechtold_downdraft_detrain_scale_m}.")

    # Joint ML physics parameterization
    parser.add_argument("--physics-parameterization", type=str, default="none",
                        choices=["none", "ml"])
    parser.add_argument("--physics-parameterization-checkpoint", type=str, default="")
    parser.add_argument("--physics-parameterization-stats", type=str, default="")
    parser.add_argument("--physics-parameterization-hidden-dim", type=int, default=128)
    parser.add_argument("--physics-parameterization-layers", type=int, default=3)
    parser.add_argument("--physics-parameterization-seed", type=int, default=0)

    # Clouds & microphysics (full-physics defaults — see the policy note above)
    parser.add_argument("--clouds", type=str, default="xu_randall",
                        choices=["none", "sundqvist", "xu_randall"])
    parser.add_argument("--cloud-rh-crit-bl", type=float, default=0.7,
                        help="Critical RH for BL cloud onset (Sundqvist). "
                             "Only active when --cloud-sigma-bl < 1.0. "
                             "Recommended ~0.55 for AMIP. Default 0.7 (disabled).")
    parser.add_argument("--cloud-sigma-bl", type=float, default=1.0,
                        help="Sigma level (p/p_s) above which rh_crit_bl applies. "
                             "Use 0.85 to cover the lowest ~1.5 km. Default 1.0 (disabled).")
    parser.add_argument("--microphysics", type=str, default="sundqvist",
                        choices=["none", "kessler", "sundqvist",
                                 "seifert_beheng", "morrison", "thompson",
                                 "p3", "sdm", "fast_sbm"])
    # Sundqvist large-scale-condensation tunables (override the SundqvistConfig
    # defaults / AIMIP-trained leaves).  These are the precipitation-efficiency
    # knobs: qc_crit is the autoconversion cloud-water threshold (rain forms only
    # for q_c >~ qc_crit; the default 5e-4 suppresses drizzle from thin clouds),
    # auto_rate the autoconversion rate, rh_crit the condensation onset RH.
    parser.add_argument("--sundqvist-qc-crit", dest="sundqvist_qc_crit",
                        type=float, default=None,
                        help="Sundqvist autoconversion cloud-water threshold "
                             "[kg/kg] (None=scheme/trained default 5e-4; bounds "
                             "1e-4..1.5e-3). Lower it to rain out thin clouds.")
    parser.add_argument("--sundqvist-rh-crit", dest="sundqvist_rh_crit",
                        type=float, default=None,
                        help="Sundqvist condensation-onset critical RH [0-1] "
                             "(None=default 0.8; bounds 0.5..1.0).")
    parser.add_argument("--sundqvist-auto-rate", dest="sundqvist_auto_rate",
                        type=float, default=None,
                        help="Sundqvist autoconversion rate c_0 [1/s] (None="
                             "default 1e-3; bounds 1e-4..1e-2).")
    parser.add_argument("--aerosol-ccn", action=argparse.BooleanOptionalAction, default=False,
                        help="Diagnose the specified cloud-droplet number "
                             "from the prescribed aerosol optical depth "
                             "(Andreae 2009 AOT-CCN inversion) instead of "
                             "the constant Nc_0.  Requires "
                             "--aerosol-forcing external and "
                             "--microphysics morrison.")
    parser.add_argument("--subgrid-autoconversion", action="store_true",
                        default=False,
                        help="Evaluate warm-rain autoconversion/accretion on "
                             "in-cloud water q_c/cf and scale by cloud fraction "
                             "(Morrison & Gettelman 2008 sub-grid closure) so "
                             "the non-linear KK2000 rate is not under-fed by "
                             "the grid-mean.  Requires --microphysics morrison.")
    parser.add_argument("--convective-precip-efficiency", type=float,
                        default=None,
                        help="Convective in-updraft precipitation efficiency "
                             "[0,1] (Tiedtke 1989 in-updraft precipitation). >0 "
                             "diverts that fraction of convective condensate "
                             "to rain (sediments via microphysics, invisible "
                             "to radiation) instead of detraining it all as "
                             "suspended cloud. Observed CPE ~0.5-0.9. Requires "
                             "--convection tiedtke or bechtold. Unset (default) "
                             "uses each scheme's own default (Tiedtke 0.0=off, "
                             "Bechtold 0.7=on, the #929 fix); pass 0.0 to force "
                             "the legacy no-split path.")
    parser.add_argument("--convective-precip-split", type=str, default="constant",
                        choices=["constant", "autoconversion"],
                        help="Convective precip-split scheme (Bechtold/Tiedtke): "
                             "'constant' uses the fixed "
                             "--convective-precip-efficiency; 'autoconversion' "
                             "derives the precip fraction PHYSICALLY from the "
                             "plume updraft cloud water (Sundqvist-1978), so the "
                             "efficiency emerges from the updraft loading instead "
                             "of a tuned constant.")
    parser.add_argument("--autoconv-q-c-crit", type=float, default=5.0e-4,
                        help="Autoconversion critical updraft cloud water [kg/kg] "
                             "for --convective-precip-split autoconversion "
                             "(Sundqvist 1978). Default 5e-4.")
    parser.add_argument("--autoconv-pe-max", type=float, default=0.9,
                        help="Ceiling on the emergent convective precip fraction "
                             "for --convective-precip-split autoconversion. "
                             "Default 0.9.")
    parser.add_argument("--convective-buoyancy-death-memory",
                        action="store_true",
                        help="Tiedtke plume buoyancy-death memory: once a "
                             "plume exhausts its cumulative buoyancy budget it "
                             "stays dead instead of reviving above an inversion "
                             "(default off lets dead plumes resume nonzero M_u "
                             "aloft, leaking convective heating to the ~100 hPa "
                             "cold point). Requires --convection tiedtke.")
    parser.add_argument("--nc-from-aerosol", action="store_true",
                        dest="aerosol_ccn",
                        help="Alias for --aerosol-ccn")

    # Topography
    parser.add_argument("--topography", type=str, default="flat")
    parser.add_argument("--topo-smoothing", type=int, default=4)
    parser.add_argument("--topo-edge-blend", type=float, default=0.3)
    parser.add_argument("--land-mask-file", type=str, default="",
                        help="Land-sea-mask NetCDF (CMIP6 sftlf / ERA5 lsm). "
                             "When set, activates the slab-land surface tile "
                             "with the land fraction from this file.")
    parser.add_argument("--albedo-land-file", type=str, default="",
                        help="Static land-albedo NetCDF (e.g. ICON-extpar ALB). "
                             "When set (with --land-mask-file), overrides the "
                             "latitude-vegetation albedo on the land tile.")
    parser.add_argument("--use-multilayer-land", default=False,
                        action=argparse.BooleanOptionalAction,
                        help="Replace the slab land tile with the differentiable "
                             "multilayer (8-layer Richards) soil column, carried in "
                             "the segment state and warm-started from the CLM "
                             "reference surface map.  Requires --land-mask-file. "
                             "--no-use-multilayer-land turns it back off when a "
                             "--config YAML enables it (e.g. for the MPAS/spectral "
                             "backends, whose standalone physics carries a passive "
                             "land tile and cannot step the soil column).")
    parser.add_argument("--multilayer-n-layers", type=int,
                        default=_EXPERIMENT_DEFAULTS.multilayer_n_layers,
                        help="Number of soil layers for --use-multilayer-land.")
    parser.add_argument("--multilayer-soil-depth", type=float,
                        default=_EXPERIMENT_DEFAULTS.multilayer_soil_depth,
                        help="Total soil-column depth [m] for --use-multilayer-land.")
    parser.add_argument("--clm-surfdata-path", type=str,
                        default=_EXPERIMENT_DEFAULTS.clm_surfdata_path,
                        help="Pre-staged CLM surfdata NetCDF (PFT/texture/glacier) "
                             "for --use-multilayer-land. Required on compute nodes "
                             "with no outbound internet (empty => download from UCAR "
                             "to /tmp, which fails there).")
    parser.add_argument("--transient-land-cover", action="store_true",
                        default=_EXPERIMENT_DEFAULTS.transient_land_cover,
                        help="Enable transient land-use/land-cover (LULC): re-weight "
                             "the multilayer land vegetation params every segment at "
                             "cover_year=start_year+elapsed/365 from "
                             "--land-cover-surfdata (soil/LAI frozen). Requires "
                             "--use-multilayer-land.")
    parser.add_argument("--land-cover-surfdata", type=str,
                        default=_EXPERIMENT_DEFAULTS.land_cover_surfdata,
                        help="Transient legoesm_surfdata NetCDF "
                             "(pft_frac(year, npft, lat, lon) in percent on the CLM5 "
                             "17-PFT axis; built by scripts/data/build_*_surfdata.py "
                             "from LUH2/HYDE/Pongratz/KK10) for "
                             "--transient-land-cover.")
    parser.add_argument("--land-ic", type=str,
                        default=_EXPERIMENT_DEFAULTS.land_ic_path,
                        help="Spun-up land IC (#746): a MultiLayerLandState "
                             "restart (.npz) from scripts/run/run_land_spinup.py. "
                             "With --use-multilayer-land, REPLACES the cold-start "
                             "soil column with the equilibrated one (avoids the "
                             "day-0 cold-start shock behind the land cold trap). "
                             "ncol/n_layers must match this run's grid.")
    parser.add_argument("--subgrid-orography-file", type=str, default="",
                        help="Subgrid orographic stddev NetCDF (ICON-extpar "
                             "SSO_STDH on a regular lat-lon grid). When set with "
                             "an orographic GWD scheme (mcfarlane/lindzen), the "
                             "per-column launch height comes from this field "
                             "(real mountains, ~0 over ocean) instead of the "
                             "scalar 500 m default.")
    parser.add_argument("--albedo-land-month", type=int, default=0,
                        help="Month (1-12) to pick from a monthly land-albedo "
                             "climatology; 0 = annual mean (default).")
    parser.add_argument("--slab-land-active", action=argparse.BooleanOptionalAction, default=False,
                        help="Activate the slab-land SEB tile using the "
                             "topography-derived land fraction (requires "
                             "--topography). No separate LSM file needed.")
    parser.add_argument("--surface-tiled", default=False,
                        action=argparse.BooleanOptionalAction,
                        help="Tiled (mosaic) surface fluxes: run --surface-bulk-scheme "
                             "(e.g. coare3) on the OCEAN tile and the fixed-roughness "
                             "land Monin-Obukhov scheme on the LAND tile, then "
                             "area-weight — instead of one scheme on the blended "
                             "surface (which runs the ocean scheme over land). "
                             "Requires --slab-land-active and --turbulence in "
                             "{louis, clubb_lite, clubb} (the kernels that consume "
                             "the injected tiled surface flux). "
                             "--no-surface-tiled turns it back off when a --config "
                             "YAML enables it (e.g. for the MPAS/spectral backends, "
                             "which do not run the tiled coupled pipeline).")
    parser.add_argument("--surface-z0-land", type=float,
                        default=_EXPERIMENT_DEFAULTS.surface_z0_land,
                        dest="surface_z0_land",
                        help="Land roughness length z0 [m] for the tiled land MOST "
                             "scheme (only used with --surface-tiled). Default "
                             f"{_EXPERIMENT_DEFAULTS.surface_z0_land}.")
    parser.add_argument("--land-soil-bucket", action="store_true", default=False,
                        dest="land_soil_bucket",
                        help="Prognostic soil-water bucket (Manabe) on the slab-land "
                             "tile: soil-moisture-limited land evaporation "
                             "(beta=beta_min+(1-beta_min)*W/W_max) instead of a "
                             "saturated wet surface everywhere. Requires "
                             "--slab-land-active.")
    parser.add_argument("--land-bucket-w-max", type=float,
                        default=_EXPERIMENT_DEFAULTS.land_bucket_w_max,
                        dest="land_bucket_w_max",
                        help="Soil-water bucket capacity [kg/m^2] (only with "
                             "--land-soil-bucket). Default "
                             f"{_EXPERIMENT_DEFAULTS.land_bucket_w_max}.")
    parser.add_argument("--land-beta-min", type=float,
                        default=_EXPERIMENT_DEFAULTS.land_beta_min,
                        dest="land_beta_min",
                        help="Minimum soil-moisture availability (dry-soil floor on "
                             "land evaporation efficiency; only with "
                             "--land-soil-bucket). Default "
                             f"{_EXPERIMENT_DEFAULTS.land_beta_min}.")
    parser.add_argument("--land-bucket-w-init-frac", type=float,
                        default=_EXPERIMENT_DEFAULTS.land_bucket_w_init_frac,
                        dest="land_bucket_w_init_frac",
                        help="Initial soil water as a fraction of W_max (only with "
                             "--land-soil-bucket). Default "
                             f"{_EXPERIMENT_DEFAULTS.land_bucket_w_init_frac}.")
    parser.add_argument("--land-k-infiltration", type=float,
                        default=_EXPERIMENT_DEFAULTS.land_K_infiltration,
                        dest="land_K_infiltration",
                        help="Saturated infiltration capacity K_s [m/s] for the "
                             "Green-Ampt infiltration-excess (Hortonian) runoff on "
                             "the soil-water bucket (only with --land-soil-bucket). "
                             f"Default {_EXPERIMENT_DEFAULTS.land_K_infiltration}.")
    parser.add_argument("--land-infil-suction-boost", type=float,
                        default=_EXPERIMENT_DEFAULTS.land_infil_suction_boost,
                        dest="land_infil_suction_boost",
                        help="Green-Ampt suction enhancement psi_f/L_f [-]: dry-soil "
                             "infiltration capacity = K_s*(1+boost) (only with "
                             "--land-soil-bucket). Default "
                             f"{_EXPERIMENT_DEFAULTS.land_infil_suction_boost}.")
    parser.add_argument("--no-land-infiltration-excess", action="store_false",
                        default=_EXPERIMENT_DEFAULTS.land_infiltration_excess,
                        dest="land_infiltration_excess",
                        help="Disable Hortonian infiltration-excess runoff on the "
                             "bucket (keep saturation excess only; all rain "
                             "infiltrates up to capacity). Default: enabled.")
    parser.add_argument("--land-stomatal-beta", action=argparse.BooleanOptionalAction, default=False,
                        dest="land_stomatal_beta",
                        help="Route the soil-water availability through the shared "
                             "land Jarvis (1976) stomatal model "
                             "(legoesm.land.stomata) instead of the bare "
                             "bucket ramp: beta=min(beta_soil, beta_canopy), closing "
                             "stomata in low light / high VPD. Requires "
                             "--land-soil-bucket.")
    parser.add_argument("--land-gs-max", type=float,
                        default=_EXPERIMENT_DEFAULTS.land_gs_max,
                        dest="land_gs_max",
                        help="Global maximum stomatal (canopy) conductance "
                             "[mol/m2/s] (StomataConfig.gs_max). Land-ET "
                             "calibration knob: gs = gs_max * f(PAR,T,VPD,soil), "
                             "so lowering it raises canopy resistance and pulls "
                             "land evapotranspiration below potential (issue "
                             "#730). Only active with --land-stomatal-beta. "
                             f"Default {_EXPERIMENT_DEFAULTS.land_gs_max} "
                             "(byte-identical when unchanged).")
    parser.add_argument("--land-soil-moisture-init-frac", type=float,
                        default=_EXPERIMENT_DEFAULTS.land_soil_moisture_init_frac,
                        dest="land_soil_moisture_init_frac",
                        help="Initial multilayer soil water as a fraction of "
                             "saturation (theta_init = frac * theta_sat) for the "
                             "cold-start (issue #730). A drier start (e.g. 0.25) "
                             "can break the over-evaporation wet loop and tip the "
                             "land into the slab-like dry attractor. Default "
                             f"{_EXPERIMENT_DEFAULTS.land_soil_moisture_init_frac} "
                             "(byte-identical when unchanged).")
    parser.add_argument("--land-surface-scheme",
                        choices=["simple_seb", "two_leaf"],
                        default=_EXPERIMENT_DEFAULTS.land_surface_scheme,
                        dest="land_surface_scheme",
                        help="Multilayer-land surface scheme (issue #730). "
                             "'simple_seb' (default) = bulk SEB with the beta_soil "
                             "moisture path; 'two_leaf' = DifferBESS two-leaf canopy "
                             "energy balance (Kelvin h_r bare-soil + two-leaf "
                             "stomatal transpiration) that holds land ET below "
                             "potential and breaks the over-evaporation wet loop. "
                             "Only affects --use-multilayer-land runs.")
    parser.add_argument("--snow-albedo-feedback", action=argparse.BooleanOptionalAction,
                        default=False, dest="snow_albedo_feedback",
                        help="Prognostic snow + snow-albedo feedback on the "
                             "slab-land tile: snow water (SWE) accumulates from "
                             "snowfall and melts (degree-day), brightening the "
                             "land albedo (snow ~0.5-0.8 vs vegetation ~0.15). "
                             "Requires an active land tile (--slab-land-active).")
    parser.add_argument("--sponge", default=False,
                        action=argparse.BooleanOptionalAction,
                        dest="sponge_enabled",
                        help="Enable the top-of-atmosphere Rayleigh sponge "
                             "(#836): damping that increases toward the model "
                             "lid to absorb upward-propagating gravity/convective "
                             "waves the hydrostatic latlon-cgrid dycore otherwise "
                             "reflects off the rigid top. Off by default. "
                             "--no-sponge turns it back off when a --config "
                             "YAML enables it (e.g. the #847 drift-lever walk).")
    parser.add_argument("--sponge-coeff-per-day", type=float, default=None,
                        dest="sponge_coeff_per_day",
                        help="Rayleigh damping rate at the model top [1/day] "
                             "(ExperimentConfig.sponge_coeff_per_day, default 2.0).")
    parser.add_argument("--sponge-sigma-top", type=float, default=None,
                        dest="sponge_sigma_top",
                        help="Sponge base: sigma below which the sin^2 damping "
                             "ramps up toward the lid (default 0.15).")
    # --cloud-conv-cloud-max closes the AMIP CLI gap for the existing
    # ExperimentConfig.cloud_conv_cloud_max field (--q-c-diagnostic / --rh-crit /
    # --subgrid-autoconv already ship from run_coupled-mirrored #647 + #613).
    parser.add_argument("--cloud-conv-cloud-max", type=float, default=None,
                        dest="conv_cloud_max",
                        help="Cap on convective (Slingo 1987) cloud cover "
                             "(CloudConfig.conv_cloud_max). Limits anvil "
                             "over-reflection. Bounds (0.1, 1.0).")
    parser.add_argument("--conv-cloud-condensate", type=float, default=None,
                        dest="conv_cloud_condensate",
                        help="In-cloud condensate [kg/kg] of the convective "
                             "anvil deck (CloudConfig.conv_cloud_condensate); "
                             "lower = optically thinner/realistic anvil. "
                             "Bounds 1e-5..1e-3.")
    parser.add_argument("--surfdata", type=str, default="",
                        help="Harmonized surface-data NetCDF "
                             "(legoesm_surfdata_*.nc). When set together with "
                             "--land-mask-file, the static land albedo is taken "
                             "from the surfdata (per-column soil-colour + PFT "
                             "vegetation blend) instead of the latitude-only "
                             "curve.")

    # Surface / diagnostics
    parser.add_argument("--monthly-means", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--t-ice-k", type=float,
                        default=constants.T_freeze_ocean,
                        help="SST floor / sea-ice ramp threshold [K]")
    parser.add_argument("--albedo-ice", type=float,
                        default=_EXPERIMENT_DEFAULTS.albedo_ice)
    parser.add_argument("--albedo-ocean", type=float,
                        default=_EXPERIMENT_DEFAULTS.albedo_ocean)
    parser.add_argument("--sfc-emissivity", type=float,
                        default=_EXPERIMENT_DEFAULTS.sfc_emissivity)
    parser.add_argument("--emissivity-ice", type=float,
                        default=_EXPERIMENT_DEFAULTS.emissivity_ice)
    parser.add_argument("--k-bl-max-per-day", type=float,
                        default=_EXPERIMENT_DEFAULTS.k_BL_max_per_day)
    parser.add_argument("--k-free-per-day", type=float,
                        default=_EXPERIMENT_DEFAULTS.k_free_per_day)

    # Moisture conservation
    parser.add_argument("--fix-moisture", action="store_true", default=False)
    # Issue #323: opt-in moist-static-energy-conserving q_v floor.  Removes
    # the latent heat of the clipped vapour sink so the per-step max(q_v, 0)
    # floor stops injecting spurious condensation heat under organised
    # convection (the kessler+sbm wind blow-up).  Default off => unchanged.
    parser.add_argument("--energy-consistent-moisture-clip",
                        action="store_true", default=False)
    # Resolved-wind moisture advection through the cdgrid dycore (issue #771).
    # OPT-IN / experimental (advective form, not discretely mass-conserving) —
    # default OFF is bit-identical to the legacy column-locked moisture path.
    _madv = parser.add_mutually_exclusive_group()
    _madv.add_argument("--moisture-advection", dest="moisture_advection",
                       action="store_true",
                       help="Opt in to resolved-wind cube moisture advection "
                            "(#771; experimental, advective form). Default off.")
    _madv.add_argument("--no-moisture-advection", dest="moisture_advection",
                       action="store_false",
                       help="Force the legacy column-locked moisture path "
                            "(the default).")
    parser.set_defaults(moisture_advection=False)

    # CMIP
    parser.add_argument("--experiment", type=str, default="")
    parser.add_argument("--start-year", type=int, default=1979)
    parser.add_argument("--forcing-update-days", type=float,
                        default=_EXPERIMENT_DEFAULTS.forcing_update_days,
                        help="Host-side forcing update cadence [days]")
    parser.add_argument("--seed", type=int, default=_EXPERIMENT_DEFAULTS.seed,
                        help="Master RNG seed for reproducibility")
    # BooleanOptionalAction so a YAML value can be turned OFF from the CLI
    # (--no-cmip-output / --no-clear-sky-diag): the clear-sky 2nd RRTMGP pass
    # roughly DOUBLES the radiation compile + per-step cost, unneeded for a
    # convection-scheme precip/albedo comparison (#872).
    parser.add_argument("--cmip-output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clear-sky-diag", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--evaluate", action="store_true", default=False,
        help="Run ClimateEval after a successful AMIP run to compare "
             "CMOR outputs against ERA5/observational reference data. "
             "Requires --cmip-output.")
    parser.add_argument(
        "--evaluation-suite", dest="evaluation_suites", nargs="+",
        default=list(_EVALUATION_DEFAULTS.suites),
        help="ClimateEval suite names to render into a single combined report. "
             "Default (unset) = ALL bundled suites (every tier); a suite whose "
             "data is missing/inapplicable is skipped + reported, not fatal. "
             "Restrict with e.g. --evaluation-suite Tier2_atmosphere_monthly.")
    parser.add_argument(
        "--evaluation-model-id", dest="evaluation_model_id", type=str,
        default=_EVALUATION_DEFAULTS.model_id,
        help="Model identifier for ClimateEval DataSourceInformation "
             f"(default: {_EVALUATION_DEFAULTS.model_id!r}).")
    parser.add_argument(
        "--evaluation-experiment-id", dest="evaluation_experiment_id", type=str,
        default=_EVALUATION_DEFAULTS.experiment_id,
        help="Experiment identifier for ClimateEval "
             f"(default: {_EVALUATION_DEFAULTS.experiment_id!r}).")
    parser.add_argument(
        "--evaluation-variant-id", dest="evaluation_variant_id", type=str,
        default=_EVALUATION_DEFAULTS.variant_id,
        help="Variant identifier for ClimateEval "
             f"(default: {_EVALUATION_DEFAULTS.variant_id!r}).")
    parser.add_argument(
        "--evaluation-data-root-dir", dest="evaluation_data_root_dir", type=str,
        default=os.environ.get(
            "LEGOESM_CLIMATEEVAL_DATA_ROOT", _EVALUATION_DEFAULTS.data_root_dir),
        help="Root directory for ClimateEval reference data (source_id/"
             "frequency/var layout). No shared canonical location — defaults "
             "from the LEGOESM_CLIMATEEVAL_DATA_ROOT env var.")
    parser.add_argument(
        "--evaluation-timerange", dest="evaluation_timerange", type=str,
        default=_EVALUATION_DEFAULTS.timerange,
        help="ClimateEval variable timerange override "
             "(e.g. '19790101/19791231'). If empty, uses the model's "
             "actual output time span.")
    parser.add_argument(
        "--evaluation-fail-missing", dest="evaluation_fail_missing",
        action="store_true",
        default=_EVALUATION_DEFAULTS.fail_on_missing_data,
        help="Fail the run if ClimateEval reference data is missing.")
    parser.add_argument(
        "--evaluation-download", dest="evaluation_download",
        action="store_true",
        default=_EVALUATION_DEFAULTS.download_missing_data,
        help="Download missing ClimateEval reference data on the fly.")
    parser.add_argument(
        "--evaluation-climateeval-python", dest="evaluation_climateeval_python", type=str,
        default=os.environ.get(
            "LEGOESM_CLIMATEEVAL_PYTHON", _EVALUATION_DEFAULTS.climateeval_python),
        help="Python interpreter of the separate, externally-installed "
             "ClimateEval environment (iris/ESMValTool; never a legoESM "
             "dependency). No hardcoded default — defaults from the "
             "LEGOESM_CLIMATEEVAL_PYTHON env var. Required (and validated "
             "to exist + be executable) when --evaluate is set.")

    # Performance
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp64", "mixed"],
                        help="Precision mode: fp32, fp64, or mixed")
    parser.add_argument("--gradient-checkpoint", action="store_true", default=False,
                        help="Enable gradient checkpointing for O(sqrt(N)) AD memory")
    parser.add_argument("--profile", type=int, default=0, metavar="N_STEPS",
                        help="Profile first N steps with jax.profiler and exit")
    parser.add_argument(
        "--production-profile", action="store_true", default=False,
        help=(
            "Bundle of conservative production-AMIP defaults aimed at "
            "long (1+ year) MPI/GPU runs.  Applied AFTER explicit user "
            "flags so any individual override still wins.  Currently "
            "raises ``--rad-update-steps`` to floor(3600/dt) (≈1-hour "
            "radiation cadence — the CESM/E3SM standard) and warns if "
            "``--fix-moisture`` is paired with prognostic-condensate "
            "microphysics (incorrect mass-budget closure).  See issue "
            "#275 for the perf rationale."
        ),
    )

    # Distributed / MPI
    parser.add_argument("--distributed", action="store_true", default=False,
                        help="Enable MPI distributed execution (auto-detected from environment)")
    parser.add_argument(
        "--distributed-mode", choices=("mpi", "spmd"),
        default=_EXPERIMENT_DEFAULTS.distributed_mode,
        help=("How multi-process runs federate (read with --distributed). "
              "'mpi' = mpi4jax halo backend (replicated cubed-sphere faces / "
              "lat-lon band / Voronoi cells; legacy default). 'spmd' = "
              "multi-controller jax.distributed: ONE global device mesh, true "
              "cubed-sphere domain decomposition (cubed-sphere only). "
              "Checkpointing + diagnostics run via gathered root-only "
              "writers. Parity receipt: 2-proc bit-exact vs "
              "single-controller, job 8686550; gate "
              "scripts/validate/validate_driver_cs_spmd_parity.py."))
    parser.add_argument(
        "--enable-tiled-dycore", action="store_true", default=False,
        help=("P4 (cube >6 devices): route the compiled segment's dynamics "
              "through the sub-face-TILED cube step (make_tiled_cc_step, "
              "(6,kt,kt) device mesh, n_devices=6*kt^2). Dynamics-only swap "
              "(physics/fixers untouched); refuses configs outside the tiled "
              "base-cut envelope. Requires --grid-type cubed_sphere."))
    parser.add_argument(
        "--enable-latlon-spmd", action="store_true", default=False,
        help=("Multi-device lat-BAND SPMD for the lat-lon C-grid dycore (A1). "
              "Requires --grid-type latlon, n_lat %% n_devices == 0. Runs the "
              "operator-split unified physics (or dynamics-only / --held-suarez) "
              "band-local. Single-process by default; add --multicontroller for "
              "the multi-node route-B lane. Distinct from --distributed (MPI)."))
    parser.add_argument(
        "--latlon-spmd-compiled-segments", action="store_true", default=False,
        help=("M2b: run each lat-lon SPMD segment as ONE compiled lax.scan "
              "(band-sharded geometry, one host dispatch per segment) instead "
              "of the per-step Python loop. Requires --enable-latlon-spmd; "
              "stateless lane only (dynamics-only / --held-suarez) — the "
              "operator-split unified-physics lane refuses it loudly. "
              "Default off = byte-identical per-step path."))
    parser.add_argument(
        "--multicontroller", action="store_true", default=False,
        help=("Promote --enable-latlon-spmd to ROUTE-B (jax.distributed, "
              "cross-process NCCL): the lat-band operator-split atm step runs "
              "one band per device across ALL processes (the multi-node lane). "
              "Requires --enable-latlon-spmd; launch under mpiexec/srun and pass "
              "--coordinator (or rely on SLURM/OMPI auto-detect)."))
    parser.add_argument(
        "--coordinator", type=str, default=None,
        help=("jax.distributed coordinator address (host:port) for "
              "--multicontroller under mpiexec (reads Open MPI OMPI_* / Cray "
              "PALS PMI_* rank env); omit for SLURM/OMPI auto-detect."))
    parser.add_argument("--ensemble-size", type=int, default=1)
    # Issue #273 follow-up: opt-in horizontal-column sharding for the
    # per-column radiation kernel.  Decouples per-column physics
    # throughput from cubed-sphere face-divisibility, unblocking
    # 4-GPU nodes (4 ∉ {1, 2, 3, 6, 24, ...}).  Requires the
    # flattened column count ``6·n·n`` divisible by the device count
    # (holds for C16/C48 production resolutions on 1–8 GPUs).
    parser.add_argument(
        "--shard-radiation-columns", action="store_true", default=False,
        help=(
            "Issue #273: shard the per-column radiation kernel across "
            "all visible devices.  Required to keep all 4 GPUs busy "
            "on a 4×A100 node where face-sharding clamps to 3."
        ),
    )
    parser.add_argument(
        "--allow-level-fallback", action="store_true", default=False,
        help=(
            "Issue #273: when the requested device count fails "
            "cubed-sphere face-sharding divisibility (e.g. 4 on a "
            "4×A100 node), route the dycore mesh to the level-"
            "parallel fallback instead of clamping to the nearest "
            "valid face-shard count (3 on a 4-GPU node).  Pair with "
            "``--shard-radiation-columns`` for the full 4-GPU "
            "unblock — dycore runs replicated on the level mesh, "
            "radiation shards columns across all 4 devices."
        ),
    )

    # Visualization
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Generate diagnostic plots after simulation completes")

    return parser


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    grid_config = GridConfig(
        grid_type=args.grid_type,
        resolution=args.resolution,
        nlev=args.nlev,
        vertical_coord=args.vertical_coord,
        p_top_Pa=args.p_top if args.p_top is not None else 200.0,
        stretching=args.stretching if args.stretching is not None else 2.0,
        use_duogrid=getattr(args, "use_duogrid", False),
    )

    dycore_config = DycoreConfig(
        discretization=args.discretization,
        dt=args.dt,
        hyperdiff_scale=args.hyperdiff_scale,
        div_damp_scale=args.div_damp_scale,
        moisture_flux_form=args.moisture_flux_form,
        mpas_nu_vert4_T=args.mpas_nu_vert4_t,
        conservation_fixer=args.conservation_fixer,
        fix_mass=args.fix_mass,
        implicit_grav_wave_use_pcg=args.implicit_grav_wave_use_pcg,
        implicit_grav_wave_damping=args.implicit_grav_wave_damping,
        # Stage 3-E: polar filter for lat-lon C-grid pole-CFL relief.
        use_polar_filter=args.use_polar_filter,
        polar_filter_cutoff_deg=args.polar_filter_cutoff_deg,
        polar_filter_max_wave_speed=args.polar_filter_max_wave_speed,
        # #836 top sponge (default OFF -> bit-identical dycore).
        sponge_coeff=args.sponge_coeff,
        sponge_width_m=args.sponge_width_m,
        sponge_shape=args.sponge_shape,
        sponge_scale_height_m=args.sponge_scale_height_m,
        # Task #25: time integrator selection.
        time_integrator=args.time_integrator,
    )

    output_config = OutputConfig(
        output_dir=args.output or "",
        diag_days=args.diag_days,
        checkpoint_days=args.checkpoint_days,
        max_wallclock_seconds=args.max_wallclock_seconds,
        monthly_means=args.monthly_means,
        cmip_output=args.cmip_output,
        clear_sky_diag=args.clear_sky_diag,
        checkpoint_format=args.checkpoint_format,
        restart_buffer_seconds=args.restart_buffer_seconds,
        evaluation=EvaluationConfig(
            enabled=args.evaluate,
            suites=tuple(args.evaluation_suites),
            model_id=args.evaluation_model_id,
            experiment_id=args.evaluation_experiment_id,
            variant_id=args.evaluation_variant_id,
            data_root_dir=args.evaluation_data_root_dir,
            fail_on_missing_data=args.evaluation_fail_missing,
            download_missing_data=args.evaluation_download,
            timerange=args.evaluation_timerange,
            climateeval_python=args.evaluation_climateeval_python,
        ),
    )

    # Preset datasets carry their own SST/SIC unit conversions (cobe SIC is
    # percent -> sic_scale=0.01; hadisst SST is Celsius -> sst_offset=T_freeze).
    # Fall back to those when the user did not pass --sst-offset/--sic-scale, so a
    # bare ``--dataset cobe`` keeps correct units; an explicit flag still wins
    # (model_driver forwards cfg.sst_offset/sic_scale into the preset config).
    try:
        from legoesm.forcing.amip import get_amip_preset
        _preset = get_amip_preset(args.dataset)
        _sst_default, _sic_default = _preset.sst_offset, _preset.sic_scale
    except ValueError:
        _sst_default, _sic_default = 0.0, 1.0

    return ExperimentConfig(
        grid=grid_config,
        dycore=dycore_config,
        output=output_config,
        days=args.days,
        start_day=args.start_day,
        dataset=args.dataset,
        forcing_path=args.forcing_path or "",
        sic_path=args.sic_path or "",
        sst_var=args.sst_var or "",
        sic_var=args.sic_var or "",
        time_var=args.time_var or "",
        lat_var=args.lat_var or "",
        lon_var=args.lon_var or "",
        sst_offset=args.sst_offset if args.sst_offset is not None else _sst_default,
        # ``is not None`` (not ``or``) so an explicit 0.0 offset / 0.0 scale — a
        # legitimate no-op-conversion or no-sea-ice sensitivity run — is not
        # silently replaced by the preset/default fallback.
        sic_scale=args.sic_scale if args.sic_scale is not None else _sic_default,
        radiation=args.radiation,
        rad_update_steps=args.rad_update_steps,
        unfused_radiation=args.unfused_radiation,
        rrtmgp_use_scan=args.rrtmgp_use_scan,
        rrtmgp_gpoint_batch_size=args.rrtmgp_gpoint_batch_size,
        rrtmgp_gpoint_checkpoint=args.rrtmgp_gpoint_checkpoint,
        rrtmgp_column_chunk_size=args.radiation_column_chunk,
        diurnal_cycle=args.diurnal_cycle,
        orbital_insolation=args.orbital_insolation,
        co2_ppmv=args.co2_ppmv,
        ch4_ppbv=args.ch4_ppbv,
        n2o_ppbv=args.n2o_ppbv,
        S_0=args.solar_s0,
        tau_equator=args.tau_equator,
        tau_pole=args.tau_pole,
        ozone_source=args.ozone_source,
        ozone_forcing=args.ozone_forcing,
        ozone_file=args.ozone_file,
        ghg_forcing=args.ghg_forcing,
        ghg_file=args.ghg_file,
        solar_source=args.solar_source,
        solar_file=args.solar_file,
        solar_tsi_var=args.solar_tsi_var,
        solar_spectral_var=args.solar_spectral_var,
        solar_spectral_band_order=args.solar_spectral_band_order,
        aerosol_forcing=args.aerosol_forcing,
        aerosol_file=args.aerosol_file,
        aerosol_reference_aod=args.aerosol_reference_aod,
        volcanic_aerosol_file=args.volcanic_aerosol_file,
        volcanic_aerosol_scale=args.volcanic_aerosol_scale,
        volcanic_aerosol_lw=args.volcanic_aerosol_lw,
        cloud_scheme=args.clouds,
        cloud_rh_crit_bl=args.cloud_rh_crit_bl,
        cloud_sigma_bl=args.cloud_sigma_bl,
        use_clubb_cloud_fraction=args.use_clubb_cloud_fraction,
        microphysics=args.microphysics,
        nc_from_aerosol=args.aerosol_ccn,
        subgrid_autoconversion=args.subgrid_autoconversion,
        convective_precip_efficiency=args.convective_precip_efficiency,
        convective_precip_split=args.convective_precip_split,
        autoconv_q_c_crit=args.autoconv_q_c_crit,
        autoconv_pe_max=args.autoconv_pe_max,
        convective_buoyancy_death_memory=args.convective_buoyancy_death_memory,
        convection=args.convection,
        turbulence=args.turbulence,
        gravity_wave_drag=args.gravity_wave_drag,
        # Tuned air-sea + cloud calibration (mirror run_coupled).
        surface_bulk_scheme=args.surface_bulk_scheme,
        surface_gustiness_zi=args.surface_gustiness_zi,
        louis_cloudtop_entrainment_efficiency=args.louis_cloudtop_entrainment_efficiency,
        surface_thermo_convention=args.bulk_thermo_convention,
        cloud_q_c_diagnostic=args.cloud_q_c_diagnostic,
        cloud_rh_crit=args.cloud_rh_crit,
        cloud_inhomogeneity_factor=args.cloud_inhomogeneity_factor,
        cloud_optics_inhomogeneity=args.cloud_optics_inhomogeneity,
        cloud_fsd=args.cloud_fsd,
        cloud_p_xr=args.cloud_p_xr,
        cloud_alpha_xr=args.cloud_alpha_xr,
        cloud_diagnostic_condensate_scheme=args.cloud_diagnostic_condensate_scheme,
        cloud_adiabatic_lwc_rate=args.cloud_adiabatic_lwc_rate,
        convective_cloud=args.convective_cloud,
        fix_moisture=args.fix_moisture,
        energy_consistent_moisture_clip=args.energy_consistent_moisture_clip,
        moisture_advection=args.moisture_advection,
        topography=args.topography,
        topo_smoothing=args.topo_smoothing,
        topo_edge_blend=args.topo_edge_blend,
        land_mask_path=args.land_mask_file,
        use_multilayer_land=args.use_multilayer_land,
        multilayer_n_layers=args.multilayer_n_layers,
        multilayer_soil_depth=args.multilayer_soil_depth,
        clm_surfdata_path=args.clm_surfdata_path,
        transient_land_cover=args.transient_land_cover,
        land_cover_surfdata=args.land_cover_surfdata,
        albedo_land_path=args.albedo_land_file,
        albedo_land_month=args.albedo_land_month,
        subgrid_orography_path=args.subgrid_orography_file,
        slab_land_active=args.slab_land_active,
        surface_tiled=args.surface_tiled,
        surface_z0_land=args.surface_z0_land,
        land_soil_bucket=args.land_soil_bucket,
        land_bucket_w_max=args.land_bucket_w_max,
        land_beta_min=args.land_beta_min,
        land_bucket_w_init_frac=args.land_bucket_w_init_frac,
        land_K_infiltration=args.land_K_infiltration,
        land_infil_suction_boost=args.land_infil_suction_boost,
        land_infiltration_excess=args.land_infiltration_excess,
        land_stomatal_beta=args.land_stomatal_beta,
        land_gs_max=args.land_gs_max,
        land_soil_moisture_init_frac=args.land_soil_moisture_init_frac,
        land_surface_scheme=args.land_surface_scheme,
        land_ic_path=args.land_ic,
        sponge_enabled=args.sponge_enabled,
        sponge_coeff_per_day=(args.sponge_coeff_per_day
                              if args.sponge_coeff_per_day is not None
                              else _EXPERIMENT_DEFAULTS.sponge_coeff_per_day),
        sponge_sigma_top=(args.sponge_sigma_top
                          if args.sponge_sigma_top is not None
                          else _EXPERIMENT_DEFAULTS.sponge_sigma_top),
        snow_albedo_feedback=args.snow_albedo_feedback,
        cloud_conv_cloud_max=args.conv_cloud_max,
        cloud_conv_cloud_condensate=args.conv_cloud_condensate,
        surfdata_path=args.surfdata,
        dynamic_albedo=args.dynamic_albedo,
        T_ice=args.t_ice_k,
        albedo_ice=args.albedo_ice,
        albedo_ocean=args.albedo_ocean,
        sfc_emissivity=args.sfc_emissivity,
        emissivity_ice=args.emissivity_ice,
        k_BL_max_per_day=args.k_bl_max_per_day,
        k_free_per_day=args.k_free_per_day,
        experiment=args.experiment,
        start_year=args.start_year,
        forcing_update_days=args.forcing_update_days,
        seed=args.seed,
        sbm_tau_c=args.sbm_tau_c,
        sbm_RH_ref=args.sbm_rh_ref,
        sbm_cape_threshold=args.sbm_cape_threshold,
        bechtold_cape_threshold=args.bechtold_cape_threshold,
        bechtold_conv_top_pa=args.bechtold_conv_top_pa,
        bechtold_downdraft_evap=args.bechtold_downdraft_evap,
        bechtold_downdraft_alpha=args.bechtold_downdraft_alpha,
        bechtold_downdraft_rh_min=args.bechtold_downdraft_rh_min,
        bechtold_downdraft_transport=args.bechtold_downdraft_transport,
        bechtold_downdraft_entrain_rate=args.bechtold_downdraft_entrain_rate,
        bechtold_downdraft_detrain_scale_m=args.bechtold_downdraft_detrain_scale_m,
        bechtold_use_ifs_cape_closure=args.bechtold_use_ifs_cape_closure,
        bechtold_use_ifs_subcloud_evap=args.bechtold_use_ifs_subcloud_evap,
        bechtold_use_ifs_inplume_precip=args.bechtold_use_ifs_inplume_precip,
        bechtold_dx_m=args.bechtold_dx_m,
        bechtold_use_ifs_downdraft=args.bechtold_use_ifs_downdraft,
        bechtold_use_ifs_shallow_closure=args.bechtold_use_ifs_shallow_closure,
        bechtold_use_ifs_capdcycl=args.bechtold_use_ifs_capdcycl,
        bechtold_use_ifs_land_rhebc=args.bechtold_use_ifs_land_rhebc,
        bechtold_use_ifs_snow_melt=args.bechtold_use_ifs_snow_melt,
        held_suarez_forcing=args.held_suarez_forcing,
        enable_latlon_spmd=args.enable_latlon_spmd,
        latlon_spmd_compiled_segments=args.latlon_spmd_compiled_segments,
        physics_parameterization=args.physics_parameterization,
        physics_parameterization_checkpoint=args.physics_parameterization_checkpoint,
        physics_parameterization_stats=args.physics_parameterization_stats,
        physics_parameterization_hidden_dim=args.physics_parameterization_hidden_dim,
        physics_parameterization_layers=args.physics_parameterization_layers,
        physics_parameterization_seed=args.physics_parameterization_seed,
        precision=args.precision,
        gradient_checkpoint=args.gradient_checkpoint,
        distributed=args.distributed,
        distributed_mode=args.distributed_mode,
        enable_tiled_dycore=args.enable_tiled_dycore,
        shard_radiation_columns=args.shard_radiation_columns,
        allow_level_fallback=args.allow_level_fallback,
        ensemble_size=args.ensemble_size,
        ic=args.ic,
        ic_path=args.ic_path,
        **({"T_init": args.t_init} if args.t_init is not None else {}),
        **({"rh_init": args.rh_init} if args.rh_init is not None else {}),
    )


def _postprocess_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    # Auto-detect MPI environment.
    # SLURM_NTASKS=1 is always set in batch jobs even for single-task GPU runs;
    # only treat it as an MPI signal when > 1 actual tasks are allocated.
    _slurm_ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    _mpi_env_vars = {"OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MPI_LOCALNRANKS"}
    # Route-B multicontroller (--multicontroller) also launches under mpiexec/srun
    # (so the SAME MPI env vars are present), but it federates via jax.distributed
    # driven by the lat-band SPMD lane — NOT the MPI/mpi4jax ``distributed`` path.
    # Its ``distributed`` MUST stay False (else config.validate_strict rejects
    # enable_latlon_spmd + distributed as mutually exclusive).
    if not args.distributed and not getattr(args, "multicontroller", False) and (
        any(key in os.environ for key in _mpi_env_vars)
        or _slurm_ntasks > 1
    ):
        args.distributed = True

    # Backward-compatible alias
    if args.radiation == "rrtmgp":
        args.radiation = "rrtmg"

    # Surfdata land albedo only takes effect when the slab-land tile is active,
    # which requires a land-sea mask.  Warn (don't fail) so the run still works
    # with the latitude-only land albedo rather than silently ignoring --surfdata.
    if args.surfdata and not args.land_mask_file:
        print("WARNING: --surfdata is ignored without --land-mask-file "
              "(the slab-land tile, and thus surfdata albedo, only activate "
              "when a land-sea mask is set); land albedo stays the "
              "latitude-only curve.")

    if (args.forcing_path is None and args.restart_from is None
            and args.dataset != "analytical"):
        parser.error("--forcing-path required (unless --dataset analytical or --restart-from)")
    if args.solar_source in ("file", "spectral_file") and not args.solar_file:
        parser.error("--solar-file required when --solar-source is file/spectral_file")
    if args.ic == "era5" and not args.ic_path:
        parser.error("--ic-path required when --ic era5")
    if args.ghg_forcing == "external" and not args.ghg_file:
        parser.error("--ghg-file required when --ghg-forcing is external")
    # Ozone/aerosol "external" with an empty path silently substitutes the
    # built-in reference climatology (use_reference_if_missing) while the run log
    # still prints the channel as ACTIVE — require the file, as solar/ghg do.
    if args.ozone_forcing == "external" and not args.ozone_file:
        parser.error("--ozone-file required when --ozone-forcing is external")
    if args.aerosol_forcing == "external" and not args.aerosol_file:
        parser.error("--aerosol-file required when --aerosol-forcing is external")
    if args.aerosol_ccn:
        if args.aerosol_forcing != "external":
            parser.error("--aerosol-ccn requires --aerosol-forcing external "
                         "(the droplet number is diagnosed from the "
                         "prescribed aerosol optical depth)")
        if args.microphysics != "morrison":
            parser.error("--aerosol-ccn requires --microphysics morrison "
                         "(the only scheme with a specified-Nc aerosol "
                         "mode)")
        # MPAS now fills the specified-Nc field via the combined-physics
        # microphysics + radiation factories (driven by the morrison
        # ``nc_from_aerosol`` switch).  Only the SPECTRAL standalone path
        # still lacks the fill.
        if args.discretization == "spectral":
            parser.error("--aerosol-ccn is not wired on the spectral "
                         "standalone path yet (the specified-Nc field is "
                         "not filled there); use cubed_sphere / latlon / "
                         "mpas, or drop --aerosol-ccn.")
    if args.subgrid_autoconversion and args.microphysics != "morrison":
        parser.error("--subgrid-autoconversion requires --microphysics "
                     "morrison (the in-cloud closure lives in the Morrison "
                     "warm-rain path)")
    _RAIN_SPLIT_SCHEMES = ("tiedtke", "bechtold", "zhang_mcfarlane",
                           "kain_fritsch", "mass_flux", "edmf")
    if (args.convective_precip_efficiency is not None
            and args.convective_precip_efficiency > 0.0
            and args.convection not in _RAIN_SPLIT_SCHEMES):
        parser.error("--convective-precip-efficiency requires a mass-flux "
                     f"convection scheme with the shared in-updraft-rain "
                     f"split: {_RAIN_SPLIT_SCHEMES}")
    if args.convective_buoyancy_death_memory and args.convection != "tiedtke":
        parser.error("--convective-buoyancy-death-memory requires --convection "
                     "tiedtke (plume buoyancy-death memory is a Tiedtke "
                     "plume-integrator option)")
    if args.dynamic_albedo and (
            args.grid_type in ("voronoi", "icosahedral", "mpas_voronoi",
                               "mpas")
            or args.discretization in ("spectral", "mpas")):
        parser.error("--dynamic-albedo is consumed by the coupled physics "
                     "pipeline (cubed_sphere / latlon only); the MPAS and "
                     "spectral standalone radiation paths use the "
                     "RRTMGPConfig constant surface albedo and would "
                     "silently ignore the flag.")
    if args.use_multilayer_land and (
            args.grid_type in ("voronoi", "icosahedral", "mpas_voronoi",
                               "mpas")
            or args.discretization in ("spectral", "mpas")):
        parser.error("--use-multilayer-land runs inside the coupled physics "
                     "pipeline (cubed_sphere / latlon only); the MPAS and "
                     "spectral standalone physics carry a PASSIVE land tile "
                     "and cannot step the soil column (the multilayer setup "
                     "crashes on the unstructured mesh: VoronoiMesh has no "
                     "lat/lat2d). Pass --no-use-multilayer-land to override "
                     "a --config YAML that enables it.")
    if args.physics_parameterization == "ml":
        if args.convection != "mass_flux" or args.turbulence != "louis":
            parser.error(
                "--physics-parameterization ml currently requires "
                "--convection mass_flux and --turbulence louis"
            )
        if not args.physics_parameterization_checkpoint or not args.physics_parameterization_stats:
            parser.error(
                "--physics-parameterization ml requires both "
                "--physics-parameterization-checkpoint and "
                "--physics-parameterization-stats"
            )
    if args.evaluate and not args.cmip_output:
        parser.error("--evaluate requires --cmip-output (ClimateEval reads "
                     "the CMOR Amon/ output tree)")

    # Auto-configure spectral runs. --truncation implies the Gaussian/spectral
    # pair, and --discretization spectral implies the Gaussian grid; an
    # explicitly conflicting choice must be a hard error, not a silent
    # override (the only silent grid fallback in the driver, audit
    # 2026-07-17). --grid-type/--discretization use a None sentinel default so
    # ANY explicit value — including one equal to the production default — is
    # distinguishable from unset and conflict-checked.
    if args.truncation is not None:
        if args.grid_type not in (None, "gaussian"):
            parser.error(
                f"--truncation implies --grid-type gaussian but "
                f"--grid-type {args.grid_type} was given; drop one of them"
            )
        if args.discretization not in (None, "spectral"):
            parser.error(
                f"--truncation implies --discretization spectral but "
                f"--discretization {args.discretization} was given; "
                "drop one of them"
            )
    if args.discretization == "spectral" and args.grid_type not in (
            None, "gaussian"):
        parser.error(
            f"--discretization spectral implies --grid-type gaussian but "
            f"--grid-type {args.grid_type} was given; drop one of them"
        )
    if args.discretization == "spectral" or args.truncation is not None:
        args.discretization = "spectral"
        args.grid_type = "gaussian"
        if args.truncation is not None:
            args.resolution = args.truncation
    # Resolve the sentinels to the production defaults AFTER the conflict
    # checks above.
    if args.grid_type is None:
        args.grid_type = "cubed_sphere"
    if args.discretization is None:
        args.discretization = "centered"

    # Canonicalise legacy ``cgrid`` → ``latlon_cgrid`` so the dycore
    # factory finds a matching (model_type, discretization, grid_type)
    # triple.  ``cdgrid`` is the cubed-sphere C-D grid; keep it as-is.
    if args.discretization == "cgrid" and args.grid_type == "latlon":
        args.discretization = "latlon_cgrid"

    # Normalise the SCVT Voronoi mesh aliases (voronoi / icosahedral /
    # mpas-as-grid_type) to the canonical ``mpas_voronoi`` before any
    # downstream consumer sees them.  See
    # ``legoesm.driver.config.normalize_grid_type``.
    from legoesm.driver.config import normalize_grid_type
    args.grid_type = normalize_grid_type(args.grid_type)

    # Issue #275 fix C: ``--production-profile`` bundles defaults that
    # the CLI cannot ship as global defaults (because they would silently
    # change behaviour for non-production callers).  Honor explicit user
    # overrides — ``--rad-update-steps`` uses ``default=None`` so we can
    # distinguish "user did not pass the flag" from "user passed
    # ``--rad-update-steps 1``".  An explicit value always wins.
    _rad_explicit = args.rad_update_steps is not None
    if args.production_profile:
        # 1-hour radiation cadence ≈ CESM/E3SM standard.  For dt < 3600
        # this means rad_update_steps = floor(3600/dt).  Only auto-set
        # when the user did not pass ``--rad-update-steps`` at all.
        if not _rad_explicit:
            rad_floor = max(1, int(3600.0 / max(args.dt, 1e-6)))
            args.rad_update_steps = rad_floor
            print(
                f"[production-profile] --rad-update-steps={rad_floor} "
                f"(≈ floor(3600/dt) for 1-hour radiation cadence)"
            )
        cadence_seconds = args.dt * args.rad_update_steps
        if cadence_seconds > 3 * 3600.0:
            print(
                f"[production-profile] WARNING: radiation cadence "
                f"{cadence_seconds:.0f}s (>3h) may smear the diurnal "
                f"cycle; consider lowering --rad-update-steps."
            )

        # Conservation closure: ``fix_moisture`` rescales only q_v and
        # ignores q_c/q_r/precipitation, so combining it with a
        # prognostic-condensate microphysics scheme silently breaks the
        # mass budget (config.py validation already warns).  Surface
        # the same warning at the AMIP CLI so production launchers
        # cannot miss it.
        _prog_microphys = args.microphysics in (
            "kessler", "sundqvist", "seifert_beheng", "morrison", "thompson",
        )
        if args.fix_moisture and _prog_microphys:
            print(
                "[production-profile] WARNING: --fix-moisture with "
                f"prognostic microphysics ({args.microphysics}) breaks "
                "total-water conservation (rescales q_v only).  Drop "
                "--fix-moisture or wait for a total-water fixer."
            )

    # Resolve the rad-update-steps default last so explicit overrides
    # and production-profile-derived values are both visible upstream.
    if args.rad_update_steps is None:
        args.rad_update_steps = 1

    return args


def _check_run_state_finite(driver) -> tuple[bool, str | None]:
    """Inspect ``driver.state`` for NaN/Inf in primary fields.

    iter-100: the standalone ``run_amip.py`` previously had no
    finiteness check.  This helper inspects the final state
    after ``driver.run(...)`` returns and reports the first
    field with a non-finite value.

    iter-104 (codex HIGH-2): the original iter-100 implementation
    only checked grid-space fields (T, u, v, p_s).  Spectral
    AMIP states (when ``run_amip.py --grid-type gaussian
    --discretization spectral``) use a different attribute
    layout — ``T_hat``, ``vor_hat``, ``div_hat``, ``lnps_hat``
    — and would silently bypass the iter-100 check (returning
    ``(True, None)``).  iter-104 extends the field list to cover
    both grid-space AND spectral attribute names; the helper
    iterates through every field name and skips the ones not
    present, so the same helper handles both layouts.

    Returns
    -------
    (ok, bad_field): tuple
        ``ok`` is False when any field in the union of
        ``{T, u, v, p_s, T_hat, vor_hat, div_hat, lnps_hat}``
        contains NaN/Inf; ``bad_field`` is the name of the
        first such field (in iteration order, grid-space first
        then spectral) or None when all present fields are
        finite.
    """
    import jax.numpy as jnp

    state = getattr(driver, "state", None)
    if state is None:
        return True, None
    # iter-104 codex HIGH-2: union of grid-space and spectral
    # field names so this helper covers both AMIP execution
    # paths.  Order: grid-space first (most common AMIP),
    # then spectral.  ``getattr(..., None)`` short-circuits
    # missing attributes.
    field_names = (
        "T", "u", "v", "p_s",
        "T_hat", "vor_hat", "div_hat", "lnps_hat",
    )
    for fname in field_names:
        f = getattr(state, fname, None)
        if f is None:
            continue
        data = getattr(f, "data", None)
        if data is None:
            continue
        try:
            ok = bool(jnp.all(jnp.isfinite(data)))
        except Exception:
            continue
        if not ok:
            return False, fname
    return True, None


def _sync_finite_verdict_across_ranks(driver, state_ok, bad_field):
    """Make the post-run finiteness verdict identical on every rank.

    Under mpi4jax topologies ``driver.state`` is the RANK-LOCAL partition, so
    ``_check_run_state_finite`` can disagree across ranks (NaN on one band
    only).  A divergent verdict means divergent ``sys.exit`` — one rank dies
    while root prints "Complete".  Reduce with logical-AND and surface the
    first bad field name from any rank.

    Multi-controller SPMD (``distributed_mode='spmd'``) needs no reduction:
    the state is globally sharded and jnp reductions are SPMD-global, so the
    verdict is process-identical by construction (mirrors the segment-boundary
    stability check in ``ModelDriver``); mpi4py there would be a second
    control plane beside jax.distributed.
    """
    _dc = getattr(driver, "_device_config", None)
    if (driver._mpi_rank is None or _dc is None
            or not getattr(_dc, "is_distributed", False)):
        return state_ok, bad_field
    from mpi4py import MPI

    state_ok = bool(MPI.COMM_WORLD.allreduce(bool(state_ok), op=MPI.LAND))
    bad_fields = MPI.COMM_WORLD.allgather(bad_field)
    bad_field = next((f for f in bad_fields if f is not None), None)
    return state_ok, bad_field


def _resolve_run_exit(run_status: str, state_ok: bool) -> int:
    """Combine the two independent failure signals into one exit code.

    0 only when the driver status is ``COMPLETED`` (via the shared
    ``status_to_exit_code`` contract) AND the final-state NaN/Inf sweep is
    clean; 1 otherwise, so a SLURM ``afterok`` chain stops instead of
    restarting from garbage.
    """
    if status_to_exit_code(run_status) != 0:
        return 1
    return 0 if state_ok else 1


def _apply_aimip_classical_overrides(
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Force the AIMIP-classical scheme set + seed it with the trained params.

    When ``--aimip-classical-checkpoint`` is given, force the classical physics
    scheme set the params were trained against — tiedtke convection, louis
    turbulence, mcfarlane GWD, xu_randall cloud, and **sundqvist microphysics**
    — and stash the trained values on ``args._aimip_params`` for the post-setup
    pipeline injection.  The cloud (xu_randall) flat fields are seeded here
    because that config is built inline in the pipeline.

    The Sundqvist microphysics is the crux of the column water budget: Tiedtke
    detrains condensate into ``q_c`` with no precipitation sink of its own, so
    ``microphysics='none'`` (the CLI default) traps column water — CWV runaway —
    and feeds unbounded ``q_c`` to the cloud optics.  Forcing the trained
    Sundqvist scheme closes the budget.  An explicit prognostic-microphysics
    override (e.g. ``--microphysics morrison``) is respected: its config is left
    untouched and the trained Sundqvist leaves are not injected (the post-setup
    block gates injection on ``args.microphysics == 'sundqvist'``).

    No-op (sets ``args._aimip_params=None``) when the flag is unset.
    """
    args._aimip_params = None
    if not getattr(args, "aimip_classical_checkpoint", None):
        return args
    import equinox as eqx
    from legoesm.training.aimip_params import AIMIPClassicalParams
    _p = eqx.tree_deserialise_leaves(
        args.aimip_classical_checkpoint, AIMIPClassicalParams.from_defaults())
    args.convection = "tiedtke"
    args.turbulence = "louis"
    args.gravity_wave_drag = "mcfarlane"
    args.clouds = "xu_randall"
    if args.microphysics == "none":
        args.microphysics = "sundqvist"
    _cc = _p.to_cloud_config()
    args.cloud_q_c_diagnostic = float(_cc.q_c_diagnostic)
    args.cloud_rh_crit = float(_cc.rh_crit)
    args._aimip_params = _p
    print(f"AIMIP-classical: forced tiedtke/louis/mcfarlane/xu_randall + "
          f"microphysics={args.microphysics} + "
          f"loaded trained params from {args.aimip_classical_checkpoint}")
    # AIMIP-classical was trained on lat-lon C-grid PE; the trained per-scheme
    # leaves inject onto the finite-volume PhysicsPipeline (cubed_sphere /
    # latlon).  The MPAS and spectral backends are off-design and differ:
    #   * spectral REFUSES tiedtke outright — the spectral run loop does not
    #     thread the physics carry yet (issue #405) and tiedtke is
    #     profile-prognostic, so _refuse_stateful_physics_unthreaded raises deep
    #     in setup BEFORE any microphysics sink runs.  Fail early + clear.
    #   * MPAS rebuilds its PhysicsConfig from config.microphysics at run(), so
    #     the schemes are forced ON (the water sink DOES close) but with DEFAULT
    #     leaves — the trained values apply only on the FV path.  Warn loudly.
    # Gate on the resolved discretization (the supported-backend signal), not the
    # grid name: an MPAS grid without --discretization mpas is rejected by the
    # driver support matrix anyway, so warning on it would be over-broad.
    _disc = getattr(args, "discretization", "centered")
    if _disc == "spectral":
        raise SystemExit(
            "AIMIP-classical forces tiedtke convection, which the spectral run "
            "loop refuses (issue #405: profile-prognostic physics carry is not "
            "threaded on spectral) — it would raise deep in setup before the "
            "microphysics sink runs. Use --grid-type cubed_sphere or latlon for "
            "the trained AIMIP-classical physics.")
    if _disc == "mpas":
        print("AIMIP-classical: WARNING — the MPAS backend rebuilds its "
              "PhysicsConfig at run() from config.microphysics; the classical "
              "schemes are forced ON (the water sink closes) but the TRAINED "
              "leaves apply ONLY on the finite-volume cubed_sphere/latlon path. "
              "Use --grid-type cubed_sphere or latlon for the trained physics.")
    return args


# Parameterization slots that an AMIP run must keep ACTIVE (never "none").
# Radiation is excluded: its choices (gray/rrtmg/rrtmgp) are all active — "none"
# is not even a legal value — so it can never be disabled.
_AMIP_REQUIRED_PHYSICS = (
    "convection", "microphysics", "turbulence", "gravity_wave_drag", "clouds",
)


def _louis_with_preserved_surface(louis_config, prev_turb_config):
    """Re-apply the run-resolved surface bulk-flux scheme onto a trained Louis.

    The AIMIP-classical ``to_louis_config()`` rebuilds its ``SurfaceLayerConfig``
    from the trained Cd/Ch/z0 at the field DEFAULTS for everything else —
    including ``bulk_scheme="constant"`` and ``gustiness_w_zi=0``.  Assigning it
    straight onto ``driver.physics.turbulence_config`` therefore CLOBBERS the
    ``--surface-bulk-scheme`` (e.g. coare3) + ``--gustiness-zi`` that
    ``_resolve_turbulence`` had already propagated into the built pipeline,
    silently reverting every AIMIP run to the constant neutral-coefficient
    surface (anemic evaporation over a calm warm ocean).

    This re-applies the previously-resolved surface ``bulk_scheme`` +
    ``gustiness_w_zi`` onto the trained Louis config, keeping the trained
    ``Cd_neutral``/``Ch_neutral``/``z0`` (the MOST/COARE schemes ignore the
    neutral ``Cd``/``Ch`` but DO use ``z0``, so preserving all three is
    correct).  No-op when there is no prior turbulence config / surface.
    """
    prev_surf = getattr(prev_turb_config, "surface", None)
    if prev_surf is None or getattr(louis_config, "surface", None) is None:
        return louis_config
    return louis_config._replace(
        surface=louis_config.surface._replace(
            bulk_scheme=prev_surf.bulk_scheme,
            gustiness_w_zi=prev_surf.gustiness_w_zi))


def _apply_sundqvist_overrides(micro_config, args):
    """Apply explicit --sundqvist-{qc-crit,rh-crit,auto-rate} overrides.

    Final precedence over both the SundqvistConfig default AND the AIMIP-trained
    leaves (so a tuning run can force the precipitation knobs).  No-op for a
    non-sundqvist microphysics, a missing config, or when no override flag is
    set — returns the SAME object so callers can detect a change by identity.
    """
    if micro_config is None or getattr(args, "microphysics", None) != "sundqvist":
        return micro_config
    overrides = {}
    if getattr(args, "sundqvist_qc_crit", None) is not None:
        overrides["qc_crit"] = args.sundqvist_qc_crit
    if getattr(args, "sundqvist_rh_crit", None) is not None:
        overrides["rh_crit"] = args.sundqvist_rh_crit
    if getattr(args, "sundqvist_auto_rate", None) is not None:
        overrides["auto_rate"] = args.sundqvist_auto_rate
    if not overrides:
        return micro_config
    return micro_config._replace(**overrides)


def _validate_sundqvist_flags(args, parser) -> None:
    """Bound-check the --sundqvist-* tunables + refuse them on backends that
    ignore the post-setup micro_config override.

    The override mutates ``driver.physics.micro_config`` (the finite-volume
    PhysicsPipeline).  MPAS and spectral rebuild ``MicrophysicsConfig`` from
    ``cfg.microphysics`` at ``run()`` and never read it, so the flags would be
    silently ignored there — refuse loudly instead.  Out-of-range values would
    silently enter the scheme (a typo like ``--sundqvist-qc-crit 1`` makes the
    tuning meaningless/unstable), so bound-check against the __param_spec__ ranges.
    """
    bounds = {"sundqvist_qc_crit": (1.0e-4, 1.5e-3),
              "sundqvist_rh_crit": (0.5, 1.0),
              "sundqvist_auto_rate": (1.0e-4, 1.0e-2)}
    set_flags = [k for k in bounds if getattr(args, k, None) is not None]
    if not set_flags:
        return
    for k in set_flags:
        lo, hi = bounds[k]
        v = getattr(args, k)
        if not (lo <= v <= hi):
            parser.error(
                f"--{k.replace('_', '-')} must be in [{lo}, {hi}], got {v}")
    disc = getattr(args, "discretization", "centered")
    grid = getattr(args, "grid_type", "")
    if disc in ("mpas", "spectral") or grid in (
            "voronoi", "icosahedral", "mpas_voronoi", "mpas"):
        parser.error(
            "--sundqvist-* overrides apply only on the finite-volume "
            "PhysicsPipeline (cubed_sphere / latlon); the MPAS and spectral "
            "backends rebuild MicrophysicsConfig at run() and would ignore them.")


def _validate_cloud_sensitivity_flags(args, parser) -> None:
    """--cloud-p-xr / --cloud-alpha-xr are valid on EVERY backend since #870
    Phase 1: the FV pipeline threads them via ``build_cloud_config`` and the
    standalone MPAS/spectral paths via ``model_driver._standalone_cloud_config``
    (which reads the same experiment fields).  The pre-#870 hard rejection on
    MPAS/spectral ("they rebuild CloudConfig at run() and would ignore them")
    is retired — that rebuild now CARRIES the override, so rejecting the flags
    there blocked a working feature with a false message (pre-merge codex
    review).  Bounds are enforced by ``ExperimentConfig.validate_strict``.
    The sundqvist micro overrides remain FV-only and keep their guard
    (``_validate_sundqvist_flags``) — those are still not threaded standalone.
    """
    return


def _require_full_physics_for_amip(args, parser) -> None:
    """Refuse an AMIP run with any parameterization slot set to ``none``.

    Policy (CLAUDE directive): an AMIP simulation is a real atmosphere, so every
    parameterization must be active.  A silently-disabled slot is a defect — e.g.
    ``microphysics='none'`` left Tiedtke's detrained condensate with no
    precipitation sink, trapping column water (CWV runaway) — so we fail LOUDLY
    rather than run an incomplete physics stack.

    Escape hatch: ``--allow-disabled-physics`` (explicit, user-owned).  The
    intrinsically-dry mode ``--held-suarez-forcing`` (Newtonian relaxation
    replaces ALL physics, radiation included) is exempt ONLY when the entire
    parameterization stack is disabled.  A MIXED run — Held-Suarez with some
    schemes active and some ``none`` (e.g. tiedtke convection with
    ``microphysics='none'`` has no precip sink) — must still pass
    ``--allow-disabled-physics``.

    ``--enable-latlon-spmd`` is NOT exempt on its own: an all-parameterization
    'none' SPMD run still leaves ``radiation`` active (AMIP cannot set radiation
    to 'none'), which the SPMD driver does not route and rejects.  A genuinely
    dry SPMD run is therefore a Held-Suarez SPMD run (``--held-suarez-forcing
    --enable-latlon-spmd``); any other disabled-slot SPMD run must opt in via
    ``--allow-disabled-physics``.
    """
    if getattr(args, "allow_disabled_physics", False):
        return
    disabled = [name for name in _AMIP_REQUIRED_PHYSICS
                if getattr(args, name, "none") == "none"]
    if not disabled:
        return
    fully_dry = len(disabled) == len(_AMIP_REQUIRED_PHYSICS)
    if fully_dry and getattr(args, "held_suarez_forcing", False):
        return
    parser.error(
        "AMIP requires every parameterization active, but these are "
        f"'none': {', '.join(disabled)}. Set them to a real scheme (the "
        "defaults already do), or pass --allow-disabled-physics for an "
        "idealized / dry-dynamics run. (Held-Suarez is exempt only when the "
        "ENTIRE stack is dry; latlon-SPMD dry runs go through Held-Suarez.)")


def main(argv: list[str] | None = None):
    # Persistent cross-process XLA compile cache (RRTMGP cold-compile ~2600 s,
    # otherwise re-paid every launch).  Idempotent; before any jit.  run_amip
    # does not go through bootstrap()/configure_backend(), so wire it directly.
    from legoesm.runtime.backend import enable_persistent_compile_cache
    enable_persistent_compile_cache()

    parser = build_arg_parser()

    # Two-pass parse so a --config file supplies defaults that explicit CLI
    # flags still override (precedence: CLI > config file > parser default).
    # Shared loader (single source of truth) — same mechanism as run_coupled.
    pre, _ = parser.parse_known_args(argv)
    if pre.config is not None:
        from legoesm.driver.run_config_yaml import load_yaml_config
        parser.set_defaults(**load_yaml_config(
            pre.config, parser,
            example_keys="'convection', 'microphysics', 'surface_bulk_scheme', "
                         "'q_c_diagnostic', 'gustiness_zi', "
                         "'bulk_thermo_convention', 'convective_cloud'"))

    args = parser.parse_args(argv)
    args = _postprocess_args(args, parser)

    # Route-B multicontroller: initialize jax.distributed BEFORE any device work
    # (ModelDriver/setup query devices; a jax op before init makes
    # jax.distributed.initialize raise "must be called before backend init").
    # The explicit --coordinator path reads OMPI_/PMI_ rank env and calls
    # jax.distributed.initialize directly (no mpi4py); --coordinator omitted
    # falls back to SLURM/OMPI auto-detect. No-op unless --multicontroller.
    if getattr(args, "multicontroller", False):
        if not getattr(args, "enable_latlon_spmd", False):
            parser.error("--multicontroller requires --enable-latlon-spmd (it "
                         "is the route-B transport for the lat-band SPMD lane).")
        from legoesm.parallel.early_init import init_multicontroller_distributed
        init_multicontroller_distributed(getattr(args, "coordinator", None))

    # --aimip-classical-checkpoint: seed the classical physics with the AIMIP
    # best-fit trained params used as INITIAL values (forces the trained scheme
    # set incl. sundqvist microphysics; stashes args._aimip_params for the
    # post-setup pipeline injection below).
    args = _apply_aimip_classical_overrides(args)

    # Enforce the full-physics policy on the FINAL resolved schemes (after the
    # AIMIP override may have promoted microphysics none -> sundqvist).
    _require_full_physics_for_amip(args, parser)
    # Bound-check the --sundqvist-* tunables + refuse them on MPAS/spectral.
    _validate_sundqvist_flags(args, parser)
    _validate_cloud_sensitivity_flags(args, parser)

    # --dt-auto: replace --dt with the ladder-validated value for this
    # (grid, resolution).  Single source of truth = the same
    # ``auto_dt_rce`` the advisory below compares against, so a
    # --dt-auto run never trips its own warning.
    if getattr(args, "dt_auto", False):
        from legoesm.driver.rce_dt import auto_dt_rce
        _ladder_dt = auto_dt_rce(args.grid_type, args.resolution)
        print(f"[run_amip] --dt-auto: {args.grid_type}/N={args.resolution} "
              f"-> dt={_ladder_dt:.0f} s (was {args.dt:.0f} s)")
        args.dt = _ladder_dt

    # iter-36: dt sanity advisory, generalising the iter-32
    # AMIP-wrapper fix to the script level. The hard-coded
    # default(--dt) = 600 was iter-13-validated only at N <= 24.
    # Higher resolutions (e.g. C48 at dt=600) BLOWUP at day 25.
    # Warn (don't raise) when the user-supplied dt exceeds the
    # iter-13/iter-26 cross-grid ladder by > 2× so the operator
    # can decide whether to override consciously.
    try:
        from legoesm.driver.rce_dt import auto_dt_rce
        try:
            _auto = auto_dt_rce(args.grid_type, args.resolution)
        except ValueError:
            # N > 96 — auto_dt_rce refuses; no comparison possible.
            _auto = None
        if _auto is not None and args.dt > 2.0 * _auto:
            print(
                f"WARNING: --dt {args.dt:.0f} s exceeds the iter-13/26 "
                f"AMIP/RCE cross-grid ladder ({_auto:.0f} s for "
                f"{args.grid_type}/N={args.resolution}) by "
                f"{args.dt / _auto:.1f}×. Long runs at this dt may "
                "BLOWUP (see CRM_implementation.md iter-12/20). "
                "Pass --dt explicitly to suppress this warning.",
                file=sys.stderr,
            )
    except ImportError:
        pass

    config = build_config_from_args(args)
    # Apply the --params calibration layer to the flattened atmosphere
    # ExperimentConfig scalar fields (issue #691).
    if getattr(args, "params", None):
        from legoesm.driver.run_config_yaml import (
            apply_params_to_config,
            build_atm_scalar_param_map,
            load_params_config,
        )
        config = apply_params_to_config(
            config, load_params_config(args.params), driver="run_amip",
            scalar_param_map=build_atm_scalar_param_map())

    from legoesm.driver.model_driver import ModelDriver

    driver = ModelDriver(config)
    print("Setup...")
    driver.setup()

    _is_root = (driver._mpi_rank is None or driver._mpi_rank == 0)

    # AIMIP-classical trained params: override the built pipeline's settable
    # scheme configs with the trained tiedtke / louis / mcfarlane values (the
    # same post-setup, pre-run() mutation the driver does for f_land / albedo;
    # captured at compile).  Cloud (xu_randall) was seeded via the flat fields
    # above (it is built inline in the pipeline).
    if getattr(args, "_aimip_params", None) is not None:
        _p = args._aimip_params
        driver.physics.convection_config = _p.to_tiedtke_config()
        # to_louis_config() rebuilds SurfaceLayerConfig at the DEFAULTS
        # (bulk_scheme="constant", gustiness_w_zi=0), so a naive assignment
        # clobbers the run-resolved --surface-bulk-scheme (e.g. coare3) +
        # --gustiness-zi that _resolve_turbulence applied to the built pipeline.
        # Preserve them so an AIMIP run honours --surface-bulk-scheme.
        driver.physics.turbulence_config = _louis_with_preserved_surface(
            _p.to_louis_config(),
            getattr(driver.physics, "turbulence_config", None))
        driver.physics.gwd_config = _p.to_mcfarlane_config()
        # Inject the trained Sundqvist microphysics leaves ONLY when the
        # resolved scheme is sundqvist (it was forced on above unless the user
        # passed a different prognostic microphysics, whose config we must not
        # clobber with sundqvist fields).  The micro_fn was wired at setup from
        # args.microphysics, so this just swaps the default leaves for trained.
        _micro_injected = False
        if args.microphysics == "sundqvist" and \
                getattr(driver.physics, "micro_config", None) is not None:
            driver.physics.micro_config = _p.to_sundqvist_config()
            _micro_injected = True
        if _is_root:
            _micro_msg = ("sundqvist (microphysics), " if _micro_injected
                          else "")
            print("AIMIP-classical trained configs injected: tiedtke (convection), "
                  "louis (turbulence), mcfarlane (GWD), " + _micro_msg +
                  "cloud via flat fields.")

    # Explicit Sundqvist precip-tunable overrides — applied LAST so they win over
    # both the scheme default and any AIMIP-trained leaves.
    if getattr(driver.physics, "micro_config", None) is not None:
        _new_micro = _apply_sundqvist_overrides(driver.physics.micro_config, args)
        if _new_micro is not driver.physics.micro_config:
            driver.physics.micro_config = _new_micro
            if _is_root:
                print(f"Sundqvist overrides: qc_crit={args.sundqvist_qc_crit} "
                      f"rh_crit={args.sundqvist_rh_crit} "
                      f"auto_rate={args.sundqvist_auto_rate}")

    if _is_root:
        _print_forcing_activity(args)

    start_step = 0
    start_day = None
    if args.restart_from:
        restart_path = Path(args.restart_from)
        if not restart_path.exists():
            print(f"ERROR: restart file not found: {restart_path}", file=sys.stderr)
            sys.exit(1)
        if _is_root:
            print(f"Loading checkpoint: {restart_path}")
        start_step, start_day = driver.load_checkpoint(restart_path)
        if _is_root:
            print(f"  Resumed at step={start_step}, day={start_day:.2f}")
        # Restore the CMOR monthly/daily accumulator state from the sidecar
        # written next to this checkpoint (setup() already rebuilt the empty
        # accumulators above), so a calendar month split across restart-chain
        # links completes — the CMOR ``Amon`` means were otherwise recreated
        # empty and lost every ~10-day link.  Keyed off the ABSOLUTE simulated
        # day load_checkpoint returned (before any --restart-start-day
        # override), matching the sidecar save name.  Runs on every rank (the
        # accumulators fill identically under SPMD); no-op when the sidecar is
        # absent (older runs) or CMOR output is off.
        if driver.diagnostics is not None:
            _cmor_sidecar = (restart_path.parent
                             / f"cmor_accum_day_{int(round(start_day)):04d}.npz")
            if (driver.diagnostics.load_cmor_accumulators(_cmor_sidecar)
                    and _is_root):
                print(f"  Restored CMOR accumulator state from "
                      f"{_cmor_sidecar.name}")
        if args.restart_start_day is not None:
            start_day = args.restart_start_day
            if _is_root:
                print(f"  start_day overridden to {start_day:.2f} (year-boundary restart)")

    if args.profile > 0:
        import jax

        profile_dir = str(Path(driver.output_dir) / "jax_profile")
        if _is_root:
            print(f"Profiling {args.profile} steps -> {profile_dir}")
        n_profile_days = args.profile * args.dt / 86400.0
        driver.config = driver.config._replace(days=int(n_profile_days + 1))
        with jax.profiler.trace(profile_dir):
            profile_status = driver.run(start_step=start_step,
                                        start_day=start_day,
                                        compiled=not args.per_step_rollout)
        if _is_root:
            print(f"Profile saved to {profile_dir}")
            print("View with: tensorboard --logdir " + profile_dir)
        # Same status→exit-code contract as the production path below: a
        # profiled blowup must not exit 0 either.
        if status_to_exit_code(profile_status) != 0:
            if _is_root:
                print(f"FAIL: profiled run did not complete cleanly "
                      f"(status={profile_status!r}).", file=sys.stderr)
            sys.exit(1)
        return

    if _is_root:
        print("Running...")
    run_status = driver.run(start_step=start_step, start_day=start_day,
                            compiled=not args.per_step_rollout)

    # Post-run FAILURE detection needs TWO independent signals — either one
    # non-clean means exit 1 (so a SLURM ``afterok`` chain STOPS instead of
    # restarting from garbage):
    #
    #  1. the driver's own run status.  ``check_stability`` flags a BLOWUP
    #     (winds / T / p_s out of *physical* bounds) at a segment boundary and
    #     returns e.g. ``"BLOWUP at day 515"``.  Those values are FINITE, so the
    #     NaN/Inf sweep below never sees them — before this, ``driver.run()``'s
    #     status was DISCARDED and a blown-up chain link printed "Complete" and
    #     exited 0, letting the chain march on (the 3-yr-chain false-completion).
    #  2. a final NaN/Inf sweep (iter-100) of the primary fields (T, u, v, p_s)
    #     — catches a non-finite state a segment-boundary probe could miss.
    #
    # ``status_to_exit_code`` is the shared status→exit-code contract.  The
    # BLOWUP verdict is already rank-identical (root checks, mpi4py bcasts —
    # see the segment-boundary stability check in ``ModelDriver``); the
    # finiteness sweep is rank-LOCAL under mpi4jax, so it is reduced across
    # ranks before the verdict so every rank exits identically.
    state_ok, bad_field = _sync_finite_verdict_across_ranks(
        driver, *_check_run_state_finite(driver))
    status_code = status_to_exit_code(run_status)
    run_failed = _resolve_run_exit(run_status, state_ok) != 0
    if _is_root:
        if not run_failed:
            print(f"Complete. Output: {driver.output_dir}")
        elif status_code != 0:
            print(
                f"FAIL: run did not complete cleanly (status={run_status!r}).  "
                f"Output (with garbage): {driver.output_dir}",
                file=sys.stderr,
            )
        else:
            print(
                f"FAIL: final state contains NaN/Inf in field "
                f"``{bad_field}``.  Output (with garbage): "
                f"{driver.output_dir}",
                file=sys.stderr,
            )
    if run_failed:
        sys.exit(1)

    if args.plot and _is_root:
        # ``plot_amip`` lives under ``scripts/plot/`` (bucket layout; see
        # tests/test_scripts_layout.py).
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plot"))
        from plot_amip import plot_amip as _plot_amip

        _plot_amip(driver.output_dir, show=False)

    if _is_root:
        from legoesm.driver.climateeval_hook import maybe_run_climateeval

        maybe_run_climateeval(config, driver.output_dir)


if __name__ == "__main__":
    main()
