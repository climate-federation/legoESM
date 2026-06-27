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
from legoesm.parallel.early_init import maybe_init_jax_distributed
maybe_init_jax_distributed()

from legoesm import constants
from legoesm.driver.config import (
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
)

_DYCORE_DEFAULTS = DycoreConfig()
_OUTPUT_DEFAULTS = OutputConfig()
_EXPERIMENT_DEFAULTS = ExperimentConfig()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AMIP simulation with prescribed SST/SIC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

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
    parser.add_argument("--grid-type", type=str, default="cubed_sphere",
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
    parser.add_argument("--discretization", type=str, default="centered",
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
        "--use-polar-filter", action="store_true",
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

    # Output
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--checkpoint-days", type=int, default=0)
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
    parser.add_argument("--radiation", type=str, default="gray",
                        choices=["gray", "rrtmg", "rrtmgp"])
    # Default is ``None`` so ``_postprocess_args`` can tell an explicit
    # ``--rad-update-steps 1`` from "the user did not pass this flag".
    # ``--production-profile`` only auto-sets the production cadence
    # when the user did not provide a value.  Resolved to ``1`` after
    # production-profile processing.
    parser.add_argument("--rad-update-steps", type=int, default=None)
    parser.add_argument("--unfused-radiation", action="store_true", default=False,
                        help="Run radiation outside the compiled segment scan")
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
    parser.add_argument("--diurnal-cycle", action="store_true", default=False)
    parser.add_argument("--dynamic-albedo", action="store_true", default=False,
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
    parser.add_argument("--turbulence", type=str, default="none",
                        choices=[
                            "none", "smagorinsky", "louis", "tke",
                            "clubb_lite", "clubb", "holtslag_boville", "ysu", "edmf",
                        ])
    parser.add_argument("--gravity-wave-drag", type=str, default="none",
                        help="GWD scheme: none, rayleigh, lindzen, mcfarlane, "
                             "hines, prognostic_spectral, ml_emulator, or a "
                             "'+'-joined composite of the diagnostic sources "
                             "(e.g. 'hines+mcfarlane' to run non-orographic + "
                             "orographic together). Validated in ExperimentConfig.")
    # Tuned air-sea + cloud knobs (the CMIP-realism calibration) — mirror
    # run_coupled so AMIP can run with the SAME tuned slab parameters. Defaults
    # (constant / 0 / None / off) keep the prior AMIP behaviour byte-identical.
    parser.add_argument("--surface-bulk-scheme", type=str, default="constant",
                        choices=["constant", "most", "coare3", "large_yeager"],
                        help="Surface-layer bulk-flux scheme (coare3 = COARE 3.0 "
                             "MOST with convective gustiness; the tuned slab value).")
    parser.add_argument("--gustiness-zi", dest="surface_gustiness_zi", type=float,
                        default=0.0,
                        help="COARE convective-gustiness BL depth z_i [m] (0=off; "
                             "tuned slab value 300).")
    parser.add_argument("--q-c-diagnostic", dest="cloud_q_c_diagnostic", type=float,
                        default=None,
                        help="In-cloud diagnostic condensate fed to radiation "
                             "[kg/kg] (None=CloudConfig default; tuned slab 3e-4).")
    parser.add_argument("--rh-crit", dest="cloud_rh_crit", type=float, default=None,
                        help="Critical RH for cloud onset (None=scheme default).")
    parser.add_argument("--convective-cloud", dest="convective_cloud",
                        action="store_true", default=False,
                        help="Add the convective (thin-cirrus) cloud-fraction "
                             "source (the tuned slab value is ON).")
    parser.add_argument("--held-suarez-forcing", action="store_true", default=False)
    parser.add_argument("--sbm-tau-c", type=float, default=7200.0)
    parser.add_argument("--sbm-rh-ref", type=float, default=0.7)
    parser.add_argument("--sbm-cape-threshold", type=float, default=70.0)

    # Joint ML physics parameterization
    parser.add_argument("--physics-parameterization", type=str, default="none",
                        choices=["none", "ml"])
    parser.add_argument("--physics-parameterization-checkpoint", type=str, default="")
    parser.add_argument("--physics-parameterization-stats", type=str, default="")
    parser.add_argument("--physics-parameterization-hidden-dim", type=int, default=128)
    parser.add_argument("--physics-parameterization-layers", type=int, default=3)
    parser.add_argument("--physics-parameterization-seed", type=int, default=0)

    # Clouds & microphysics
    parser.add_argument("--clouds", type=str, default="none",
                        choices=["none", "sundqvist", "xu_randall"])
    parser.add_argument("--cloud-rh-crit-bl", type=float, default=0.7,
                        help="Critical RH for BL cloud onset (Sundqvist). "
                             "Only active when --cloud-sigma-bl < 1.0. "
                             "Recommended ~0.55 for AMIP. Default 0.7 (disabled).")
    parser.add_argument("--cloud-sigma-bl", type=float, default=1.0,
                        help="Sigma level (p/p_s) above which rh_crit_bl applies. "
                             "Use 0.85 to cover the lowest ~1.5 km. Default 1.0 (disabled).")
    parser.add_argument("--microphysics", type=str, default="none",
                        choices=["none", "kessler", "sundqvist",
                                 "seifert_beheng", "morrison", "thompson",
                                 "p3", "sdm", "fast_sbm"])
    parser.add_argument("--aerosol-ccn", action="store_true", default=False,
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
                        default=0.0,
                        help="Tiedtke convective precipitation efficiency "
                             "[0,1] (1989 in-updraft precipitation). >0 "
                             "diverts that fraction of convective condensate "
                             "to rain (sediments via microphysics, invisible "
                             "to radiation) instead of detraining it all as "
                             "suspended cloud. Observed CPE ~0.5-0.9. Requires "
                             "--convection tiedtke.")
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

    # Surface / diagnostics
    parser.add_argument("--monthly-means", action="store_true", default=False)
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

    # CMIP
    parser.add_argument("--experiment", type=str, default="")
    parser.add_argument("--start-year", type=int, default=1979)
    parser.add_argument("--forcing-update-days", type=float,
                        default=_EXPERIMENT_DEFAULTS.forcing_update_days,
                        help="Host-side forcing update cadence [days]")
    parser.add_argument("--seed", type=int, default=_EXPERIMENT_DEFAULTS.seed,
                        help="Master RNG seed for reproducibility")
    parser.add_argument("--cmip-output", action="store_true", default=False)
    parser.add_argument("--clear-sky-diag", action="store_true", default=False)

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
        "--enable-latlon-spmd", action="store_true", default=False,
        help=("Single-process multi-device lat-BAND SPMD for the lat-lon C-grid "
              "dycore (A1). Requires --grid-type latlon, n_lat %% n_devices == 0, "
              "and dynamics-only or --held-suarez physics (stateful physics not "
              "yet SPMD-routed). Distinct from --distributed (MPI)."))
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
        p_top_Pa=args.p_top or 200.0,
        stretching=args.stretching or 2.0,
        use_duogrid=getattr(args, "use_duogrid", False),
    )

    dycore_config = DycoreConfig(
        discretization=args.discretization,
        dt=args.dt,
        hyperdiff_scale=args.hyperdiff_scale,
        div_damp_scale=args.div_damp_scale,
        conservation_fixer=args.conservation_fixer,
        fix_mass=args.fix_mass,
        implicit_grav_wave_use_pcg=args.implicit_grav_wave_use_pcg,
        implicit_grav_wave_damping=args.implicit_grav_wave_damping,
        # Stage 3-E: polar filter for lat-lon C-grid pole-CFL relief.
        use_polar_filter=args.use_polar_filter,
        polar_filter_cutoff_deg=args.polar_filter_cutoff_deg,
        polar_filter_max_wave_speed=args.polar_filter_max_wave_speed,
        # Task #25: time integrator selection.
        time_integrator=args.time_integrator,
    )

    output_config = OutputConfig(
        output_dir=args.output or "",
        diag_days=args.diag_days,
        checkpoint_days=args.checkpoint_days,
        monthly_means=args.monthly_means,
        cmip_output=args.cmip_output,
        clear_sky_diag=args.clear_sky_diag,
        checkpoint_format=args.checkpoint_format,
        max_wallclock_seconds=args.max_wallclock_seconds,
        restart_buffer_seconds=args.restart_buffer_seconds,
    )

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
        sst_offset=args.sst_offset or 0.0,
        sic_scale=args.sic_scale or 1.0,
        radiation=args.radiation,
        rad_update_steps=args.rad_update_steps,
        unfused_radiation=args.unfused_radiation,
        rrtmgp_use_scan=args.rrtmgp_use_scan,
        rrtmgp_gpoint_batch_size=args.rrtmgp_gpoint_batch_size,
        rrtmgp_gpoint_checkpoint=args.rrtmgp_gpoint_checkpoint,
        diurnal_cycle=args.diurnal_cycle,
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
        microphysics=args.microphysics,
        nc_from_aerosol=args.aerosol_ccn,
        subgrid_autoconversion=args.subgrid_autoconversion,
        convective_precip_efficiency=args.convective_precip_efficiency,
        convective_buoyancy_death_memory=args.convective_buoyancy_death_memory,
        convection=args.convection,
        turbulence=args.turbulence,
        gravity_wave_drag=args.gravity_wave_drag,
        # Tuned air-sea + cloud calibration (mirror run_coupled).
        surface_bulk_scheme=args.surface_bulk_scheme,
        surface_gustiness_zi=args.surface_gustiness_zi,
        cloud_q_c_diagnostic=args.cloud_q_c_diagnostic,
        cloud_rh_crit=args.cloud_rh_crit,
        convective_cloud=args.convective_cloud,
        fix_moisture=args.fix_moisture,
        energy_consistent_moisture_clip=args.energy_consistent_moisture_clip,
        topography=args.topography,
        topo_smoothing=args.topo_smoothing,
        topo_edge_blend=args.topo_edge_blend,
        land_mask_path=args.land_mask_file,
        albedo_land_path=args.albedo_land_file,
        albedo_land_month=args.albedo_land_month,
        subgrid_orography_path=args.subgrid_orography_file,
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
        held_suarez_forcing=args.held_suarez_forcing,
        enable_latlon_spmd=args.enable_latlon_spmd,
        physics_parameterization=args.physics_parameterization,
        physics_parameterization_checkpoint=args.physics_parameterization_checkpoint,
        physics_parameterization_stats=args.physics_parameterization_stats,
        physics_parameterization_hidden_dim=args.physics_parameterization_hidden_dim,
        physics_parameterization_layers=args.physics_parameterization_layers,
        physics_parameterization_seed=args.physics_parameterization_seed,
        precision=args.precision,
        gradient_checkpoint=args.gradient_checkpoint,
        distributed=args.distributed,
        shard_radiation_columns=args.shard_radiation_columns,
        allow_level_fallback=args.allow_level_fallback,
        ensemble_size=args.ensemble_size,
        ic=args.ic,
        ic_path=args.ic_path,
        **({"T_init": args.t_init} if args.t_init is not None else {}),
        **({"rh_init": args.rh_init} if args.rh_init is not None else {}),
    )


def _postprocess_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    # Auto-detect MPI environment
    if not args.distributed and any(
        key in os.environ for key in (
            "OMPI_COMM_WORLD_SIZE", "PMI_SIZE",
            "SLURM_NTASKS", "MPI_LOCALNRANKS",
        )
    ):
        args.distributed = True

    # Backward-compatible alias
    if args.radiation == "rrtmgp":
        args.radiation = "rrtmg"

    if (args.forcing_path is None and args.restart_from is None
            and args.dataset != "analytical"):
        parser.error("--forcing-path required (unless --dataset analytical or --restart-from)")
    if args.solar_source in ("file", "spectral_file") and not args.solar_file:
        parser.error("--solar-file required when --solar-source is file/spectral_file")
    if args.ic == "era5" and not args.ic_path:
        parser.error("--ic-path required when --ic era5")
    if args.ghg_forcing == "external" and not args.ghg_file:
        parser.error("--ghg-file required when --ghg-forcing is external")
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
    if args.convective_precip_efficiency > 0.0 and args.convection != "tiedtke":
        parser.error("--convective-precip-efficiency requires --convection "
                     "tiedtke (only Tiedtke implements in-updraft "
                     "precipitation)")
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

    # Auto-configure spectral runs
    if args.discretization == "spectral" or args.truncation is not None:
        args.discretization = "spectral"
        args.grid_type = "gaussian"
        if args.truncation is not None:
            args.resolution = args.truncation

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


def main(argv: list[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args = _postprocess_args(args, parser)

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

    from legoesm.driver.model_driver import ModelDriver

    driver = ModelDriver(config)
    print("Setup...")
    driver.setup()

    _is_root = (driver._mpi_rank is None or driver._mpi_rank == 0)

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
            driver.run(start_step=start_step, start_day=start_day)
        if _is_root:
            print(f"Profile saved to {profile_dir}")
            print("View with: tensorboard --logdir " + profile_dir)
        return

    if _is_root:
        print("Running...")
    driver.run(start_step=start_step, start_day=start_day)

    # iter-100: post-run finiteness check.  Pre-iter-100,
    # ``run_amip.py`` had ZERO blowup detection (``grep -c
    # isfinite`` = 0 in 450 lines).  A NaN-producing AMIP run
    # would silently complete and print "Complete." while
    # writing garbage to the output directory.  iter-98's audit
    # of the OMIP/atmosphere-matrix BLOWUP-reporting bug flagged
    # this as a separate gap; iter-100 closes it.
    #
    # The check inspects the final ``driver.state`` for NaN/Inf
    # in the primary atmospheric fields (T, u, v, p_s).  When
    # non-finite, the run is flagged FAIL with a clear message
    # and the script exits with code 1 so wrappers
    # (``run_amip_cross_grid.sh``) can detect failure.
    state_ok, bad_field = _check_run_state_finite(driver)
    if _is_root:
        if state_ok:
            print(f"Complete. Output: {driver.output_dir}")
        else:
            print(
                f"FAIL: final state contains NaN/Inf in field "
                f"``{bad_field}``.  Output (with garbage): "
                f"{driver.output_dir}",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.plot and _is_root:
        # ``plot_amip`` lives under ``scripts/plot/`` (bucket layout; see
        # tests/test_scripts_layout.py).
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plot"))
        from plot_amip import plot_amip as _plot_amip

        _plot_amip(driver.output_dir, show=False)


if __name__ == "__main__":
    main()
