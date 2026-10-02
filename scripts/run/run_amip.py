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
    CLUBB_SCALAR_FIELDS,
    VALID_RADIATION,
    ZM_SCALAR_FIELDS,
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
from legoesm.grids.halo import CORNER_FILL_MODES

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
    # The gaussian/spectral standalone radiation path integrates with the
    # configured constant S_0 — the solar FILE (TSI + 14-band spectral) is
    # not threaded there (same gap the CMIP6 deck labels; keep the two
    # tables telling the same truth).  The MPAS lane threads it since the
    # 2026-07-23 port: daily-sampled traced forcing["tsi"] (+
    # ["solar_spectral_fraction"] under --solar-source spectral_file with
    # rrtmg/rrtmgp) into the standalone radiation.
    _grid = getattr(args, "grid_type", None) or "cubed_sphere"
    _disc = getattr(args, "discretization", None) or ""
    solar_file_unthreaded = (_grid == "gaussian" and _disc == "spectral")

    def _flag(active: bool) -> str:
        return "ACTIVE" if active else "inert  (gray radiation)"

    print("[run_amip] Forcing-channel activity for this run:")
    print("  SST/SIC                              ACTIVE        (radiation-independent)")
    if solar_file_active and solar_file_unthreaded:
        print("  Solar TSI                            inert         "
              f"({_disc or _grid} path uses constant S_0; solar file not threaded)")
    elif solar_file_active:
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
                        choices=["sigma", "hybrid", "cam_l32"],
                        help="cam_l32 = CAM6's 32-level hybrid table (nlev must be 32)")
    parser.add_argument("--p-top", type=float, default=None)
    parser.add_argument("--stretching", type=float, default=None)
    parser.add_argument("--transition-exponent", type=int, default=None,
                        choices=[2, 3],
                        help="hybrid B(eta)=eta**n exponent. 3 (default) carries "
                             "NEGATIVE layer mass below 664 hPa, i.e. above ~3450 m "
                             "of orography -- 0.92%% of the planet by area. 2 moves "
                             "that to ~498 hPa / ~5870 m and covers all of ETOPO, at "
                             "the cost of different level placement (#1029).")
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
    # "fv3_duo" is the certified FV3 six-face duo-cube fv_dynamics lane
    # (cubed_sphere grid only; dry dynamics, fp64, nlev in {5, 10};
    # the only physics is --held-suarez-forcing on the hydrostatic arm —
    # the component factory refuses everything else loudly).
    parser.add_argument("--discretization", type=str, default=None,
                        choices=["centered", "finite_volume", "cgrid",
                                  "latlon_cgrid", "cdgrid", "mpas", "spectral",
                                  "fv3_duo"])
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
    # #1028: persistent D-grid winds on the cubed-sphere hydrostatic lane.
    # The default cell-centre path interpolates the prognostic winds to the
    # D-grid corners and back on EVERY step; that outer projection is a
    # measured eddy damper (paired 200-day C36 Held-Suarez: max wind
    # 13.0 -> 38.6 m/s sigma, 12.6 -> 40.9 hybrid, one knob, PR #1462).
    parser.add_argument(
        "--persistent-dgrid", action=argparse.BooleanOptionalAction,
        default=False,
        help="Carry the cube's prognostic winds in FV3 D staggering between "
             "steps instead of projecting cell-centre -> corner -> cell-centre "
             "every step (#1028).  Cubed-sphere hydrostatic, single process; "
             "every other lane refuses it rather than silently running the "
             "damped path.",
    )
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
    # #1029 ω-side: SB81 α-weighted κT·ω/p conversion (hybrid latlon lane).
    parser.add_argument(
        "--sb81-omega-conversion", action="store_true",
        default=_DYCORE_DEFAULTS.sb81_omega_conversion,
        help="Use the SB81 α-weighted energy conversion (ω/p dynamic part) "
             "on the hybrid lat-lon C-grid — discretization-consistent with "
             "the geopotential and ln p^SB gradients (#1029). Default OFF: "
             "the consistent form unmasks the #1029(b) lid-wave instability "
             "sooner (held_suarez_topo blowup day ~49 -> ~12); opt-in until "
             "the lid treatment lands.",
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
                 "ssp_rk54", "ssp_rk54_scan", "rk4",
                 "leapfrog", "leapfrog_si"],
        help="Time integrator.  'auto' (default) selects each dycore's "
             "own stable default: ssp_rk3 on cube/lat-lon (IEEE-"
             "identical to existing runs) and ssp_rk54_scan on MPAS "
             "(the biharmonic hyperdiffusion eigenvalues at production "
             "dt fall outside ssp_rk3's stability region — the former "
             "'hidden CFL' blow-up).  An explicit name is forwarded "
             "verbatim to every dycore, including ssp_rk3 on MPAS for "
             "deliberate integrator-sensitivity runs.  ssp_rk3_scan is "
             "the JIT-compile-time optimised variant for production "
             "lat-lon C-grid AMIP.  leapfrog / leapfrog_si are the "
             "SPECTRAL dycore's single-physics-eval path — required to run "
             "PROGNOSTIC physics (clubb_lite TKE / bechtold / prognostic "
             "GWD) faithfully on gaussian/spectral (#405); the per-RK-stage "
             "ssp_rk3 spectral path still diagnostic-swaps prognostic "
             "schemes.",
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
    parser.add_argument("--diag-days", type=float, default=5.0,
                        help="Diagnostic/blow-up-check cadence in days. Values "
                             "below 1 are honoured and are how a blow-up gets "
                             "localised in time: the reported failure day is the "
                             "first SAMPLE, not the first bad step.")
    parser.add_argument("--corner-fill", dest="corner_fill",
                        choices=CORNER_FILL_MODES,
                        default=_DYCORE_DEFAULTS.corner_fill,
                        help="Cubed-sphere cube-vertex halo corner fill "
                             "(inert on other grids). avg = 2-point average.")
    parser.add_argument("--hyperdiff-scale", type=float,
                        default=_DYCORE_DEFAULTS.hyperdiff_scale,
                        help="Dycore hyperdiffusion multiplier")
    parser.add_argument("--a-h-scale", type=float,
                        default=_DYCORE_DEFAULTS.a_h_scale,
                        dest="a_h_scale",
                        help="Second-order Laplacian viscosity multiplier "
                             "(A_h = a_h_scale * 3e-3 * dx_min^2 / dt). NOT "
                             "scale-selective: it damps as k^2, so it reaches "
                             "the baroclinic eddies that drive the "
                             "midlatitude jet. At 2.5 deg / dt=75 s the "
                             "default 1.0 damps a 2000 km wave in 0.73 d and "
                             "4000 km in 2.9 d, comparable to or faster than "
                             "the ~1-2 d eddy growth time — the failure mode "
                             "component_factory records at 16x this value "
                             "(\'crushing the midlatitude eddy-driven "
                             "jets\'). 0 relies on the scale-selective "
                             "4th-order hyperdiff alone.")
    parser.add_argument("--k-h-scale", dest="k_h_scale", type=float,
                        default=None,
                        help="Separate scale for horizontal THERMAL diffusivity "
                             "K_h (None = follow --a-h-scale, byte-identical). "
                             "Lets momentum viscosity be reduced for the "
                             "eddy-driven-jet response while keeping the "
                             "thermal smoothing that suppresses vertical "
                             "computational modes.")
    parser.add_argument("--mpas-nu-vert4-t", type=float,
                        default=_DYCORE_DEFAULTS.mpas_nu_vert4_T,
                        help="MPAS vertical biharmonic hyperdiffusion of T "
                             "[1/s] — #930 2Δσ vertical-checkerboard cure "
                             "(0 disables)")
    parser.add_argument("--mpas-conservative-tracer-clamp",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="MPAS floors: borrow the clipped negative tracer "
                             "deficit back from the positive cells in the same "
                             "column instead of the mass-CREATING plain "
                             "max(q,0).  The naive clamp invents ~+30 kg/m2/yr "
                             "of water on a century AMIP run (measured); the "
                             "borrow cuts that 10.4x.  ON by default (owner "
                             "decision 2026-08-16: conserving form always); "
                             "--no-mpas-conservative-tracer-clamp restores the "
                             "legacy clamp for bit-comparison runs.")
    parser.add_argument("--mpas-vert-advection-scheme",
                        choices=("upwind", "van_leer", "sb"),
                        default=_DYCORE_DEFAULTS.mpas_vert_advection_scheme,
                        help="Vertical advection scheme on the MPAS sigma lane "
                             "(theta, tracers, edge winds).  'upwind' (default) "
                             "is first-order donor cell, whose implicit "
                             "diffusion K_sigma=|sigma_dot|*dsigma/2 warms the "
                             "tropical UTLS by +0.822 K/day (measured, 91.4 hPa). "
                             "'van_leer' is the 2nd-order TVD alternative "
                             "(bounded face reconstruction; monotone update "
                             "under a Courant condition nu_k+nu_k+1<=1, "
                             "derived for a UNIFORM grid -- measured max "
                             "0.0642 on that run, 15.6x inside it; needs "
                             "nlev>=4).  Default = bit-identical.")
    parser.add_argument("--mpas-sponge-del2-top-layers", type=int, default=None,
                        dest="mpas_sponge_del2_top_layers",
                        help="CAM-style top diffusion sponge: number of top layers whose "
                             "del2 viscosity is enhanced (MPAS; 0 = off)")
    parser.add_argument("--mpas-sponge-del2-top-factor", type=float, default=None,
                        dest="mpas_sponge_del2_top_factor",
                        help="top-layer del2 viscosity multiplier of the sponge, ramping "
                             "geometrically to 1 below the sponge layers (1.0 = off)")
    parser.add_argument("--mpas-div-damp4-scale", type=float, default=None,
                        dest="mpas_div_damp4_scale",
                        help="Divergence-SELECTIVE biharmonic damping on the MPAS "
                             "hydrostatic lane, as a multiple of CAM-FV's ldiv4 "
                             "nondimensional rate 0.01*area^2/dt (0 = off). 1.0 is "
                             "CAM's rate per APPLICATION, not CAM's behaviour: CAM "
                             "applies it per acoustic substep with a per-cell "
                             "coefficient, so treat this as a calibration knob. "
                             "Unlike --div-damp-scale / the del2+del4 viscosities, this "
                             "damps only the curl-free mode and leaves balanced flow alone.")
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
    parser.add_argument("--cmip-resolution-deg", type=float,
                        default=_OUTPUT_DEFAULTS.cmip_resolution_deg,
                        help="Lat-lon spacing [deg] of the CMOR output grid. "
                             "Must track the MESH: at --resolution 4 the native "
                             "spacing is 379 km and 5 deg output is matched, but "
                             "a finer mesh written at 5 deg throws the "
                             "refinement away, and the tropical rain band -- one "
                             "to two cells wide -- becomes unscorable. "
                             f"Default {_OUTPUT_DEFAULTS.cmip_resolution_deg}.")
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
    parser.add_argument("--physics-update-steps", type=int, default=1,
                        help="Run the whole column physics every N steps and "
                             "re-apply its cached tendencies in between (CAM "
                             "cadence; MPAS lane). rad-update-steps must be a "
                             "multiple. 1 = every step.")
    parser.add_argument("--cld-macmic-num-steps", type=int, default=1,
                        help="CAM6 cld_macmic_num_steps: sub-cycle turbulence "
                             "(macrophysics) + microphysics N times at "
                             "dt_phys/N inside each physics step, after the "
                             "convective increment (MPAS lane). 1 = parallel "
                             "split.")
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
                        choices=["standard", "analytical", "mls", "none"])
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
                        choices=["constant", "coare3", "large_yeager",
                                 "large_yeager_cesm"],
                        help="Surface-layer bulk-flux scheme (coare3 = COARE 3.0 "
                             "MOST with convective gustiness; the tuned slab value). "
                             "Matches ExperimentConfig.validate_strict — 'most' is "
                             "not an accepted AMIP surface scheme (coare3 is the "
                             "MOST-with-gustiness variant).")
    parser.add_argument("--surface-stability-scheme", type=str,
                        dest="surface_stability_scheme", default="dyer1974",
                        choices=["dyer1974", "beljaars_holtslag1991",
                                 "grachev2007_sheba", "gryanik2020"],
                        help="Stable-branch (zeta>0) Monin-Obukhov similarity "
                             "functions for the surface layer (bulk_flux psi_m/"
                             "psi_h). dyer1974 (default) is byte-identical: "
                             "the short-tail -5*zeta on most/large_yeager, but "
                             "on coare3 (the AMIP surface scheme) the "
                             "COARE-native stable form, which is already the "
                             "long-tail BH91 fit — so beljaars_holtslag1991 is "
                             "a rounding-level change on coare3, and the "
                             "genuinely different strong-stable tails there "
                             "are grachev2007_sheba/gryanik2020.")
    parser.add_argument("--hb-kvf-min", dest="hb_kvf_min", type=float,
                        default=None,
                        help="Free-atmosphere diffusivity floor override "
                             "[m^2/s] for turbulence schemes carrying kvf_min "
                             "(holtslag_boville; scheme default 0.01). "
                             "Causality probe for the polar-night stable-"
                             "transport runaway; None keeps the scheme "
                             "default byte-identically.")
    parser.add_argument("--surface-z-ref-model-level", dest="surface_z_ref_model_level",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Tell the ocean MOST solver the real height of the lowest "
                             "model level instead of labelling its inputs as z_ref (10 m). "
                             "Unset keeps the scheme's own value (True, the production "
                             "default); --no-surface-z-ref-model-level turns it off.")
    parser.add_argument("--surface-ocean-q-sfc-saline", dest="surface_ocean_q_sfc_saline",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Ocean surface humidity = 0.98 x q_sat(SST, p_s) (sea water at "
                             "the surface pressure) instead of fresh water at the lowest level.")
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
    parser.add_argument("--clubb-cf-override-strength",
                        dest="cloud_clubb_cf_override_strength", type=float,
                        default=None,
                        help="Blend strength [0,1] toward diagnostic-CLUBB cloud "
                             "fraction in the BL when --use-clubb-cloud-fraction "
                             "is on (1.0=full replacement, which drove a real-SST "
                             "surface-heating runaway; ~0.3-0.5 gentler+stable). "
                             "None=CloudConfig default (1.0).")
    parser.add_argument("--clubb-cf-override-floor",
                        dest="cloud_clubb_cf_override_floor", type=float,
                        default=None,
                        help="Minimum BL cloud fraction [0,1] the CLUBB override "
                             "may leave — breaks the cloud->0 cloud-temperature "
                             "runaway that full reduction caused, so a LARGER "
                             "albedo fix can run stably (~0.15-0.25). None="
                             "CloudConfig default (0.0 = no floor).")
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
    parser.add_argument("--cloud-vertical-overlap-optics",
                        dest="cloud_vertical_overlap_optics",
                        choices=["none", "max_random", "mcica"], default="none",
                        help="VERTICAL cloud-overlap optics. The solver has "
                             "no McICA/overlap, so cloud spread thinly over "
                             "many partly cloudy layers is solved as ONE "
                             "deep uniform cloud. 'max_random' re-solves "
                             "the column as --cloud-n-subcolumns "
                             "deterministic maximum-random-overlap "
                             "subcolumns and averages: measured -30%% cloud "
                             "albedo and +18 W/m2 OLR. Costs n_sub x the "
                             "radiation time. 'mcica' (CAM6) gives each "
                             "g-point its own maximum-random subcolumn: "
                             "one solve, sampling noise per g-point. "
                             "Mutually exclusive with "
                             "--cloud-partial-coverage-optics=two_column.")
    parser.add_argument("--cloud-n-subcolumns", dest="cloud_n_subcolumns",
                        type=int, default=8,
                        help="Subcolumns for --cloud-vertical-overlap-optics"
                             "=max_random. Measured against a Monte-Carlo "
                             "reference: 8 leaves 2.5%% of the signal, 4 "
                             "leaves 18%%. Default 8.")
    parser.add_argument("--cloud-partial-coverage-optics",
                        dest="cloud_partial_coverage_optics",
                        choices=["none", "two_column"], default="none",
                        help="Partial-cloud-COVER optics. The radiation "
                             "solver has no McICA/overlap: it sees ONE "
                             "homogeneous column at the grid-mean water "
                             "path, R(cf*tau_ic), which is ALWAYS brighter "
                             "than the independent-column cf*R(tau_ic)+"
                             "(1-cf)*R(0) because R is concave. "
                             "'two_column' thins the path by the exact "
                             "inversion of that identity (chi<=1, so it can "
                             "only DIM). 'none'=legacy, byte-identical.")
    parser.add_argument("--cloud-saturation-scheme",
                        dest="cloud_saturation_scheme",
                        choices=["liquid", "mixed_phase"],
                        default="mixed_phase",
                        help="Saturation curve for the cloud-fraction RH. "
                             "'mixed_phase' (DEFAULT since 2026-09-17) blends "
                             "liquid/ice saturation by the scheme's own "
                             "condensate ice-fraction ramp (IFS alpha(T) "
                             "convention); warm cloud is unchanged. 'liquid' "
                             "(legacy, byte-identical) measures RH against "
                             "liquid (Tetens) saturation at every temperature, "
                             "so ice-saturated air reads RH well below "
                             "rh_crit and the RH cloud schemes diagnose NO "
                             "cloud where the model carries ice: ~0.55-0.75 in "
                             "the TTL/anvil (#1521) and 0.662 at 230 K / "
                             "900 hPa, which left the February Arctic with "
                             "0.0 %% cover against 35-40 %% observed.")
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
    parser.add_argument("--clubb-trop-cloud-top-press", type=float, default=None,
                        dest="clubb_trop_cloud_top_press",
                        help="CLUBB upper domain limit [Pa] (CAM "
                             "trop_cloud_top_press): mixing tapered to zero "
                             "above it. Default: the scheme's own 0 = no limit.")
    for _f, _leaf in {**ZM_SCALAR_FIELDS, **CLUBB_SCALAR_FIELDS}.items():
        _cls = ("ZhangMcFarlaneConfig" if _f.startswith("zm_")
                else "CLUBBParams")
        parser.add_argument("--" + _f.replace("_", "-"), type=float,
                            default=None, dest=_f,
                            help=f"{_cls}.{_leaf} (legal range: the scheme's "
                                 "__param_spec__). Default: the scheme's own "
                                 "value.")
    parser.add_argument("--clubb-q-flux-scale", type=float, default=None,
                        dest="clubb_q_flux_scale",
                        help="Moisture-only multiplier on CLUBB's q_v eddy "
                             "diffusivity at the faces whose sigma lies in "
                             "--clubb-q-flux-scale-sigma-band (cloud-base "
                             "mixing probe). Default: the scheme's own 1.0.")
    parser.add_argument("--clubb-q-flux-scale-sigma-band", nargs=2, type=float,
                        default=None, dest="clubb_q_flux_scale_sigma_band",
                        metavar=("LO", "HI"),
                        help="Sigma band (lo hi) of faces --clubb-q-flux-scale "
                             "acts on; required with it.")
    parser.add_argument("--clubb-prognostic", dest="clubb_prognostic",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Run CLUBB as a PROGNOSTIC higher-order closure: "
                             "the scheme carries 15 higher-order moments as "
                             "real state, including the total-water variance "
                             "and its covariance with temperature, instead of "
                             "re-estimating them each step from a mixing "
                             "length times a local gradient. This is the "
                             "sub-grid variance the cloud PDF otherwise has to "
                             "guess. Requires --turbulence clubb. Default off "
                             "= the diagnostic path (byte-identical).")
    parser.add_argument("--clubb-liquid-partition", dest="clubb_liquid_partition",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Let CLUBB exchange CLOUD LIQUID with the host, as "
                             "CAM does: the closure's total water carries the "
                             "existing cloud water in, and its own diagnosed "
                             "liquid is written back to the condensate tracer "
                             "instead of being returned as vapour. Without it "
                             "the host takes its cloud FRACTION from CLUBB and "
                             "its cloud WATER from a tracer CLUBB never wrote, "
                             "and the two disagree. Requires --turbulence clubb "
                             "and --clubb-prognostic. Default off = the "
                             "historical bridge (byte-identical).")
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
    parser.add_argument("--snow-age-activation-K",
                        dest="snow_age_activation_K", type=float, default=None,
                        help="Snow grain-growth activation temperature [K] for "
                             "the BATS/CLM temperature-dependent snow-age clock "
                             "(BATS uses 5000; bounds 0..20000). The age clock "
                             "then accumulates dt*exp(A*(1/T_freeze - 1/T_snow)), "
                             "so cold dry polar snow keeps its fresh albedo "
                             "while melting snow darkens as before. "
                             "None = the LandAlbedoConfig default (5000 K); 0 = "
                             "calendar clock.")
    parser.add_argument("--land-soil-freeze-thaw", dest="land_soil_freeze_thaw",
                        action=argparse.BooleanOptionalAction,
                        default=_EXPERIMENT_DEFAULTS.land_soil_freeze_thaw,
                        help="Soil-water freeze/thaw (latent zero-curtain) in "
                             "the multilayer land, as in CLM5. Default off "
                             "(sensible-only). Requires --use-multilayer-land.")
    parser.add_argument("--land-snow-tau-days", dest="land_snow_tau_days",
                        type=float, default=_EXPERIMENT_DEFAULTS.land_snow_tau_days,
                        help="Snow-albedo age e-folding time [days]. Default: "
                             "150 (production, 2026-09-15); the land "
                             "calibration's own value is 3.674 d. A 3.7-day "
                             "clock darkens any snowpack older than a few weeks "
                             "to its minimum albedo regardless of temperature, "
                             "which is why every polar cell measured 0.521 "
                             "against an observed 0.70-0.82. Pairs with "
                             "--snow-age-activation-K, which alone does not "
                             "move it.")
    parser.add_argument("--cloud-cap-floor", dest="cloud_cap_floor_on",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="ATTRIBUTION LEVER (Arctic self-isolation A/B, arm 1): "
                             "hand radiation a cloud floor poleward of "
                             "--cloud-cap-floor-lat-deg below --cloud-cap-floor-p-max-pa "
                             "(cloud fraction >= --cloud-cap-floor-cf, grid-mean liquid path "
                             ">= cf * --cloud-cap-floor-q-c * dp/g). Radiation-only; "
                             "prognostic condensate and diagnostics untouched. Off = production.")
    parser.add_argument("--cloud-cap-floor-lat-deg", dest="cloud_cap_floor_lat_deg",
                        type=float, default=None, help="[deg] None = scheme default 70")
    parser.add_argument("--cloud-cap-floor-p-max-pa", dest="cloud_cap_floor_p_max_pa",
                        type=float, default=None, help="[Pa] floor applies below this; None = 70000")
    parser.add_argument("--cloud-cap-floor-cf", dest="cloud_cap_floor_cf",
                        type=float, default=None, help="imposed cloud fraction; None = 0.8")
    parser.add_argument("--cloud-cap-floor-q-c", dest="cloud_cap_floor_q_c",
                        type=float, default=None, help="[kg/kg] imposed in-cloud liquid; None = 5e-5")
    for _nm, _lo, _hi, _dv in (("rhmini", 0.5, 0.99, 0.80), ("rhmaxi", 1.0, 1.1, 1.0),
                               ("rhminis", 0.85, 1.0, 1.0), ("rhmaxis", 1.0, 1.1, 1.0)):
        parser.add_argument(f"--cloud-cam6-{_nm}", dest=f"cloud_cam6_{_nm}",
                            type=float, default=None,
                            help=f"CAM6 ice-stratus ramp cldfrc2m {_nm} "
                                 f"(cloud scheme cam6_clubb; bounds {_lo}..{_hi}); "
                                 f"None = {_dv} (CAM6 CLUBB default)")
    parser.add_argument("--cloud-cover-condensate-q-ref",
                        dest="cloud_cover_condensate_q_ref", type=float,
                        default=None,
                        help="Condensate-aware cover floor for the RH cloud "
                             "schemes: cf >= q_cond/(q_cond + q_ref) [kg/kg], "
                             "so layers carrying prognostic condensate are "
                             "never clear to radiation (bounds 1e-6..1e-3; "
                             "LOWER => more cover per unit condensate). "
                             "None = scheme default 0.0 (off, byte-identical).")
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
    parser.add_argument(
        "--fv3-duo-windows", type=int, default=None, metavar="KT",
        help=("fv3_duo window SPMD: split every face into KT x KT windows, one "
              "per device (6*KT*KT ranks, --distributed --distributed-mode "
              "spmd). Explicit only; the device count must match exactly. "
              "Requires --fv3-duo-window-pad. Default: the face layout."))
    parser.add_argument(
        "--fv3-duo-window-pad", type=int, default=None, metavar="PAD",
        help=("Window halo width for --fv3-duo-windows (a measured per-deck "
              "value, e.g. 11 at C48 with 3 acoustic substeps; no default)."))
    parser.add_argument(
        "--fv3-duo-column-lane", action="store_true", default=False,
        help=("fv3_duo as a COLUMN model inside the MPAS lane (route A): the "
              "duo is the dynamics operator of the CAM6-suite loop through "
              "FV3DuoColumnModel; physics runs on (nCells, nlev) columns "
              "unchanged. Six faces, fp64, hydrostatic; checkpoints/ERA5 IC "
              "not yet (M4/M5)."))
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
    parser.add_argument("--bechtold-subsidence-solve", type=str,
                        choices=["implicit_flux", "advective"],
                        default=_EXPERIMENT_DEFAULTS.bechtold_subsidence_solve,
                        dest="bechtold_subsidence_solve",
                        help="Bechtold compensating-subsidence vertical solve: "
                             "implicit_flux (conservative, default) or advective "
                             "(legacy; truncation-order conservation only — the "
                             "stability escape hatch of the 2026-07-22 day-65 "
                             "blowup bisect). "
                             f"Default {_EXPERIMENT_DEFAULTS.bechtold_subsidence_solve}.")
    parser.add_argument("--bechtold-rain-vapor-sink", type=str,
                        choices=["formation", "vapour_mass"],
                        default=_EXPERIMENT_DEFAULTS.bechtold_rain_vapor_sink,
                        dest="bechtold_rain_vapor_sink",
                        help="Where the in-plume convective rain's vapour is "
                             "debited and its latent heat released: formation "
                             "(default, at the rain-formation levels) or "
                             "vapour_mass (legacy whole-column spread by vapour "
                             "mass; the A/B control). "
                             f"Default {_EXPERIMENT_DEFAULTS.bechtold_rain_vapor_sink}.")
    parser.add_argument("--zm-land-fraction", type=str,
                        choices=["required", "none"],
                        default=_EXPERIMENT_DEFAULTS.zm_land_fraction,
                        dest="zm_land_fraction",
                        help="Zhang-McFarlane column land fraction: required "
                             "(the run must supply one; it picks the land/"
                             "ocean autoconversion coefficient) or none "
                             "(explicit aquaplanet, ocean coefficients "
                             "everywhere). "
                             f"Default {_EXPERIMENT_DEFAULTS.zm_land_fraction}.")
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
    parser.add_argument("--bechtold-m-b-max", dest="bechtold_M_b_max", type=float,
                        default=_EXPERIMENT_DEFAULTS.bechtold_M_b_max,
                        help="Bechtold cloud-base mass-flux cap [kg/m^2/s], range "
                             "0.02-0.15 (production 0.05 since 2026-09-16; the "
                             "old 0.02 fallback bound 71 %% of tropical columns).")
    parser.add_argument("--bechtold-enable-cmt", dest="bechtold_enable_cmt",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Gregory-1997 convective momentum transport in Bechtold "
                             "(applied to the MPAS edge winds).")
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
                        choices=["none", "sundqvist", "xu_randall",
                                 "cam6_clubb"])
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
    parser.add_argument("--hard-saturation-adjustment",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Opt-in iterated hard saturation adjustment "
                             "for the warm-rain schemes (kessler/seifert_beheng/"
                             "morrison/thompson/p3): where q_v exceeds "
                             "hard_sat_adjust_threshold * q_sat (default 1.1), "
                             "drain q_v onto the liquid saturation curve "
                             "(conserving c_pd*T + L_v*q_v), rate-limited to "
                             "hard_sat_max_heating_K per step (default 5 K ~ "
                             "2 g/kg), removing local super-saturation pools the "
                             "smooth path cannot.  Applied POST-STEP on the MPAS "
                             "path (the validated placement) and in-scheme on "
                             "the spectral/coupled path.  Default off (moist "
                             "path byte-identical).  The threshold + cap default "
                             "to the scheme-config values (matching the "
                             "validated configuration); override via "
                             "--hard-sat-adjust-threshold / "
                             "--hard-sat-max-heating-k or --params.")
    parser.add_argument("--hard-sat-adjust-threshold", type=float, default=None,
                        dest="hard_sat_adjust_threshold",
                        help="Override the hard-saturation-adjustment RH "
                             "trigger (drain where q_v > threshold * q_sat). "
                             "Requires --hard-saturation-adjustment; bounds "
                             "[1, 2] per the scheme __param_spec__ (scheme "
                             "default 1.1).")
    parser.add_argument("--hard-sat-max-heating-k", type=float, default=None,
                        dest="hard_sat_max_heating_K",
                        help="Override the hard-saturation-adjustment per-step "
                             "latent-heating cap [K]. Requires "
                             "--hard-saturation-adjustment; bounds [0.5, 50] "
                             "per the scheme __param_spec__ (scheme default "
                             "5 K; 10 K = the day-137 summer-regime tuning "
                             "arm).")
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
    parser.add_argument("--convective-rain-to-surface",
                        action=argparse.BooleanOptionalAction, default=True,
                        dest="convective_rain_to_surface",
                        help="Route the in-updraught convective rain that survives "
                             "the scheme's own sub-cloud evaporation straight to "
                             "surface precipitation (IFS convention) instead of "
                             "into the microphysics rain tracer, where it is "
                             "re-evaporated at grid-mean humidity.")
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
    parser.add_argument("--land-model", choices=("none", "slab", "multilayer"),
                        default=None,
                        help="Select the land surface model BY NAME, the same "
                             "three names the coupled driver and run_lmip_smoke "
                             "already use (CoupledESMConfig.land_mode): 'none' "
                             "= NO land surface model at all (land temperature "
                             "falls back to the neighbouring prescribed SST "
                             "minus a lapse rate, and there is no soil, no "
                             "water store and no stomatal control), 'slab' = "
                             "the slab surface-energy-balance tile, "
                             "'multilayer' = slab plus the Richards multilayer "
                             "soil and CLM texture/PFT maps. Sets "
                             "--slab-land-active and --use-multilayer-land for "
                             "you; passing this together with either of those "
                             "is an error rather than a silent override. "
                             "Omitting it leaves those two flags in charge, so "
                             "existing decks are unaffected.")
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
    parser.add_argument("--land-update-seconds", type=float,
                        default=_EXPERIMENT_DEFAULTS.land_update_seconds,
                        dest="land_update_seconds",
                        help="Multilayer-land call interval [s]; 0 = every "
                             "host step (legacy). A positive value calls the "
                             "tile every round(interval/dt) steps on the "
                             "interval-MEAN forcing and holds its fluxes in "
                             "between (same shape as the hourly radiation "
                             "cadence).")
    parser.add_argument("--multilayer-n-layers", type=int,
                        default=_EXPERIMENT_DEFAULTS.multilayer_n_layers,
                        help="Number of soil layers for --use-multilayer-land.")
    parser.add_argument("--multilayer-soil-depth", type=float,
                        default=_EXPERIMENT_DEFAULTS.multilayer_soil_depth,
                        help="Total soil-column depth [m] for --use-multilayer-land.")
    parser.add_argument("--land-calibrated-physics",
                        action=argparse.BooleanOptionalAction,
                        default=_EXPERIMENT_DEFAULTS.land_calibrated_physics,
                        dest="land_calibrated_physics",
                        help="Run the multilayer land tile in the SAME model its "
                             "baked per-PFT tables were calibrated under "
                             "(legoesm.land.config.calibrated_multilayer_setup): "
                             "MOST surface exchange, SimpleSEB, Farquhar stomata on "
                             "a prescribed carbon state, and the calibration soil "
                             "column. Without it the baked canopy conductance "
                             "(Vc_max25/g1/LCMA) is inert and the tables run under "
                             "land physics they were never fitted to. Requires "
                             "--use-multilayer-land --land-stomatal-beta "
                             "--land-surface-scheme simple_seb and the calibration "
                             "soil column (checked, never silently overridden).")
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
    parser.add_argument("--land-interface-flux", dest="land_interface_flux",
                        choices=["legacy_dual", "unified"],
                        default=_EXPERIMENT_DEFAULTS.land_interface_flux,
                        help="Flux law the slab-land SEB debits at the "
                             "land-air interface. 'legacy_dual' (default, "
                             "byte-identical): the slab uses its own constant-"
                             "C_H/C_E no-stability bulk law while the "
                             "atmosphere debits the turbulence scheme's "
                             "stability-dependent surface fluxes — two laws, "
                             "measured same-state mismatch +75..+152 W/m2. "
                             "'unified': the slab consumes the SAME surface-"
                             "layer law the atmosphere applies (ONE flux law "
                             "at the interface; the semi-implicit "
                             "max(dF/dT,0)*dT_skin discretization term "
                             "remains and is only small at a short "
                             "--rad-update-steps cadence). Requires an "
                             "active --turbulence scheme and an active slab "
                             "land tile.")
    parser.add_argument("--c-land", dest="C_land", type=float,
                        default=_EXPERIMENT_DEFAULTS.C_land,
                        help="Slab-land effective heat capacity [J/m2/K] "
                             "(validate_strict bounds [1e4, 1e8]). Default "
                             f"{_EXPERIMENT_DEFAULTS.C_land:.1e} (~0.15 m "
                             "active soil layer).")
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
    parser.add_argument("--land-soil-init", type=str,
                        default=_EXPERIMENT_DEFAULTS.land_soil_init,
                        choices=["aridity", "saturation_fraction"],
                        dest="land_soil_init",
                        help="How the multilayer soil is seeded at a cold start. "
                             "'aridity' (default) maps the initial atmosphere's "
                             "near-surface relative humidity into the plant-"
                             "available range, capping the start at FIELD "
                             "CAPACITY and leaving "
                             "--land-soil-moisture-init-frac inert. "
                             "'saturation_fraction' uses that fraction times "
                             "porosity instead, so the soil can start near "
                             "SATURATION — use it to keep a run out of the "
                             "dry-soil attractor, at the cost of the desert "
                             "evaporation runaway the aridity seed prevents.")
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
                        choices=["simple_seb", "two_leaf", "clm_ml"],
                        default=_EXPERIMENT_DEFAULTS.land_surface_scheme,
                        dest="land_surface_scheme",
                        help="Multilayer-land surface scheme. 'two_leaf' "
                             "(DEFAULT) is the two-leaf canopy: it partitions each "
                             "cell into canopy and soil and gives each its own "
                             "resistance, which is what reproduces observed latent "
                             "heat and photosynthesis. 'simple_seb' is for ACADEMIC "
                             "/ SIMPLIFIED tests only — with stomata off it "
                             "evaporates at potential, and with them on it throttles "
                             "BARE GROUND with a conductance that has no leaf-area "
                             "dependence. 'clm_ml' is the full multilayer canopy.")
    parser.add_argument("--clm-ml-use-surfdata-pft", action=argparse.BooleanOptionalAction,
                        default=_EXPERIMENT_DEFAULTS.clm_ml_use_surfdata_pft,
                        dest="clm_ml_use_surfdata_pft",
                        help="CLM-ML only: give each column its DOMINANT PFT (argmax of "
                             "the surface map's pft_fractions) instead of one pft_clm for "
                             "all columns -> mixed-PFT heterogeneous columns, compiled at "
                             "O(#distinct structures) by the group-by-structure canopy "
                             "scan. Opt-in (default off = single pft_clm). Only affects "
                             "--land-surface-scheme clm_ml + --use-multilayer-land.")
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
    parser.add_argument("--mpas-land-lapse-k-per-km", type=float, default=None,
                        dest="mpas_land_lapse_K_per_km",
                        help="MPAS lane only: lapse-adjust the LAND fraction's "
                             "surface-temperature anchor by this rate [K/km] "
                             "times elevation (the AMIP loader fills land "
                             "cells with nearest-ocean sea-level SST, which "
                             "overheats elevated terrain). 0=off (default); "
                             "6.5=ICAO standard atmosphere.")
    parser.add_argument("--mpas-land-beta", type=float, default=None,
                        dest="mpas_land_beta",
                        help="MPAS lane only: land evaporation efficiency in "
                             "[0, 1] throttling the land-fraction surface "
                             "humidity gradient (1.0=saturated wet swamp, "
                             "default; ~0.6 first-order continental mean).")
    parser.add_argument("--mpas-land-beta-soil",
                        action=argparse.BooleanOptionalAction, default=False,
                        dest="mpas_land_beta_soil",
                        help="MPAS lane only (#1312 phase 2b): thread the "
                             "interactive multilayer land's per-cell "
                             "root-zone beta_soil into the turbulence "
                             "surface humidity (traced forcing['beta_land'], "
                             "one-step lag) — the land latent flux is then "
                             "throttled by the soil's own moisture state, "
                             "REPLACING the static --mpas-land-beta over "
                             "land. Requires --use-multilayer-land.")
    parser.add_argument("--mpas-land-stress-from-land",
                        action=argparse.BooleanOptionalAction, default=False,
                        dest="mpas_land_stress_from_land",
                        help="MPAS lane only: surface stress over the land "
                             "fraction from the land model (canopy roughness, "
                             "rho u*^2) instead of the atmosphere's bulk "
                             "(ocean-roughness) call; heat fluxes unchanged. "
                             "Requires --mpas-land-beta-soil.")
    parser.add_argument("--mpas-land-params-refresh",
                        action=argparse.BooleanOptionalAction, default=True,
                        dest="mpas_land_params_refresh",
                        help="MPAS lane, multilayer two-leaf land: rebuild "
                             "LAI, canopy height and soil albedo from the "
                             "surfdata climatology every land step, as the "
                             "offline calibration does (default on). "
                             "--no-mpas-land-params-refresh keeps the start "
                             "day's parameters for the whole run. Acts only "
                             "with --use-multilayer-land.")
    parser.add_argument("--mpas-qv-smooth-del2-m2s", type=float, default=None,
                        dest="mpas_qv_smooth_del2_m2s",
                        help="MPAS lane only: horizontal q_v del2 (unweighted "
                             "SCVT Laplacian) smoothing diffusivity [m^2/s], "
                             "applied post-step + a q>=0 floor (~1e5-1e6 "
                             "typical at 240 km; 0=off, default). Conserves "
                             "(to roundoff) the per-level mixing-ratio integral "
                             "but NOT column water vapour (non-conservative "
                             "filter). "
                             "Setup refuses coefficients above the explicit "
                             "monotonicity bound for the mesh+dt.")
    parser.add_argument("--mpas-qv-smooth-del4-m4s", type=float, default=None,
                        dest="mpas_qv_smooth_del4_m4s",
                        help="MPAS lane only: horizontal q_v del4 (biharmonic) "
                             "smoothing diffusivity [m^4/s], applied post-step "
                             "alongside the del2 (0=off, default). "
                             "Scale-selective: its damping ratio between any "
                             "two scales is the del2's SQUARED (measured 6.08x "
                             "vs 2.47x between 240 and 479 km on the "
                             "subdivision-6 mesh), so it holds grid-scale "
                             "noise down without flattening the resolved "
                             "humidity gradients. NOT monotone, so the q>=0 "
                             "floor can fire; setup refuses coefficients above "
                             "the explicit stability bound for the mesh+dt.")
    parser.add_argument("--hines-total-rms-wind", type=float, default=None,
                        dest="hines_total_rms_wind",
                        help="Hines (1997) non-orographic GWD launch RMS wind "
                             "[m/s] (default 2.0). Larger = stronger "
                             "non-orographic drag; the low-level extratropical "
                             "westerly bias is the observable lever.")
    parser.add_argument("--hines-launch-p", type=float, default=None,
                        dest="hines_launch_p",
                        help="Hines non-orographic GWD LAUNCH PRESSURE [Pa] "
                             "(e.g. 70000 = 700 hPa). Unset/0 keeps the legacy "
                             "SURFACE launch, where the wave is born "
                             "supersaturated in the weakly stratified boundary "
                             "layer (sigma_sat = N/m_star is smallest there) "
                             "and breaks at its own launch level instead of "
                             "aloft. No drag is deposited at or below the "
                             "launch level.")
    parser.add_argument("--hines-fmax", type=float, default=None,
                        dest="hines_Fmax",
                        help="Hines saturation momentum-flux cap [Pa] "
                             "(default 0.1).")
    parser.add_argument("--e3sm-cam-source", type=str, default=None,
                        dest="e3sm_cam_source",
                        help="E3SM CAM gravity-wave source: orographic | "
                             "frontal | convective | background, or a "
                             "'+'-joined set (CAM6 f09: "
                             "orographic+frontal+convective). Validated by "
                             "ExperimentConfig.validate_strict.")
    parser.add_argument("--e3sm-cam-effgw-cm", type=float, default=None,
                        dest="e3sm_cam_effgw_cm",
                        help="Frontal-source efficiency (CAM effgw_cm; CAM6 "
                             "f09 1.0). Default None = --e3sm-cam-effgw.")
    parser.add_argument("--e3sm-cam-effgw-beres", type=float, default=None,
                        dest="e3sm_cam_effgw_beres",
                        help="Beres convective-source efficiency (CAM "
                             "effgw_beres_dp; CAM6 f09 0.4). Default None = "
                             "--e3sm-cam-effgw.")
    parser.add_argument("--e3sm-cam-frontgfc", type=float, default=None,
                        dest="e3sm_cam_frontgfc",
                        help="Frontogenesis threshold for the frontal source "
                             "[K^2/(m^2 s)] (CAM frontgfc; CAM6 f09 3.0e-15). "
                             "Default None = kernel default 1.25e-15.")
    parser.add_argument("--e3sm-cam-beres-variant", type=str, default=None,
                        choices=["e3sm", "cam6"],
                        dest="e3sm_cam_beres_variant",
                        help="Beres source kernel oracle: e3sm (default) or "
                             "cam6 (gw_convect.F90: end-off spectrum shift, "
                             "real storm speed, interface source level).")
    parser.add_argument("--e3sm-cam-dttke-intrinsic",
                        action=argparse.BooleanOptionalAction, default=None,
                        dest="e3sm_cam_dttke_intrinsic",
                        help="Spectral GW heating form: intrinsic-frequency "
                             "sum (c-u)*gwut (CAM6 gw_common.F90:690) vs the "
                             "E3SM-3.0.1 ground-relative sum c*gwut (default).")
    parser.add_argument("--e3sm-cam-mfcc-table", type=str, default=None,
                        dest="e3sm_cam_mfcc_table_path",
                        help="Offline Beres lookup table netcdf (CAM "
                             "gw_drag_file, newmfspectra40_dc25.nc). Unset = "
                             "the documented analytic stand-in spectrum.")
    parser.add_argument("--e3sm-cam-pgwv", type=int, default=None,
                        dest="e3sm_cam_pgwv",
                        help="Number of gravity-wave phase-speed bins.")
    parser.add_argument("--e3sm-cam-effgw", type=float, default=None,
                        dest="e3sm_cam_effgw",
                        help="Gravity-wave efficiency factor (dimensionless).")
    parser.add_argument("--e3sm-cam-taubgnd", type=float, default=None,
                        dest="e3sm_cam_taubgnd",
                        help="Uniform background spectrum amplitude; total "
                             "absolute launch flux ~ taubgnd*sqrt(pi)*c0/dc "
                             "[Pa].")
    parser.add_argument("--e3sm-cam-c0", type=float, default=None,
                        dest="e3sm_cam_c0",
                        help="Width of source spectrum in phase speed [m/s].")
    parser.add_argument("--e3sm-cam-launch-p", type=float, default=None,
                        dest="e3sm_cam_launch_p",
                        help="Launch pressure level for the spectrum [Pa].")
    parser.add_argument("--e3sm-cam-latitude-taper",
                        action=argparse.BooleanOptionalAction, default=None,
                        dest="e3sm_cam_latitude_taper",
                        help="cos(lat) taper of the E3SM frontal/background drag "
                             "toward the poles (E3SM structured-dycore branch); "
                             "--no-... is the unstructured/MPAS branch.")
    parser.add_argument("--mcfarlane-tau-max", type=float, default=None,
                        dest="mcfarlane_tau_max",
                        help="McFarlane orographic GWD surface stress cap [Pa] "
                             "(default 10.0). Reaches the kernel on every lane "
                             "via gwd_config_for.")
    parser.add_argument("--mcfarlane-k-wave", type=float, default=None,
                        dest="mcfarlane_k_wave",
                        help="McFarlane orographic GWD horizontal wavenumber "
                             "[1/m] (default 2*pi/100 km). Scales the launch "
                             "stress tau_0 ~ G_0*rho*N*k*h^2*U. No "
                             "__param_spec__ entry yet (bounds undecided), so "
                             "this flag is its ONLY route -- --params cannot "
                             "reach it.")
    parser.add_argument("--homogeneous-ice-nucleation",
                        action=argparse.BooleanOptionalAction, default=False,
                        dest="homogeneous_ice_nucleation",
                        help="Morrison only: Koop/Ren-MacKenzie homogeneous "
                             "cirrus ice nucleation, which pins RH over ice "
                             "near 1.45-1.6 instead of letting it run away "
                             "(the model reached 288%% at 228 K). Treats the "
                             "cause of the supersaturation pile-up rather "
                             "than draining it after the fact.")
    parser.add_argument("--morrison-warm-rain-scheme",
                        choices=["kk2000", "kk2000_cam6", "seifert_beheng",
                                 "seifert_beheng_sb2001"],
                        default=ExperimentConfig._field_defaults[
                            "morrison_warm_rain_scheme"],
                        dest="morrison_warm_rain_scheme",
                        help="Morrison warm-rain autoconversion/accretion "
                             "law (kk2000 = Khairoutdinov-Kogan 2000).")
    parser.add_argument("--morrison-autocon-fact", type=float,
                        default=ExperimentConfig._field_defaults[
                            "morrison_autocon_fact"],
                        dest="morrison_autocon_fact",
                        help="Multiplier on the kk2000 autoconversion rate "
                             "(CAM6 MG2-style; 1.0 = unscaled, <1 keeps "
                             "more cloud liquid). kk2000 only.")
    parser.add_argument("--morrison-accre-enhan-fact", type=float,
                        default=ExperimentConfig._field_defaults[
                            "morrison_accre_enhan_fact"],
                        dest="morrison_accre_enhan_fact",
                        help="Multiplier on kk2000 accretion of cloud by "
                             "rain (MG2 accre_enhan; 1.0 = unscaled). "
                             "kk2000 only.")
    parser.add_argument("--morrison-warm-rain-incloud",
                        action=argparse.BooleanOptionalAction,
                        default=ExperimentConfig._field_defaults[
                            "morrison_warm_rain_incloud"],
                        dest="morrison_warm_rain_incloud",
                        help="CAM6 MG2 in-cloud warm rain: autoconversion "
                             "and accretion on cloud water divided by CAM6's "
                             "ast = max(CLUBB liquid, aist ice) cloud "
                             "fraction, tendencies scaled back by it. "
                             "Needs --turbulence clubb, cloud scheme "
                             "cam6_clubb and "
                             "cld_macmic_num_steps>=2.")
    parser.add_argument("--morrison-flavor", choices=["mg", "sam"],
                        default="mg", dest="morrison_flavor",
                        help="Morrison parameter flavor: 'mg' (E3SM/CESM "
                             "Morrison-Gettelman set, GCM default) or 'sam' "
                             "(SAM/gSAM M2005 set — faster cloud-ice fall "
                             "fall_b_i=0.865 and PSD bounds tuned against "
                             "anvil-ice over-accumulation, the exact disease "
                             "of the 2026-07 AMIP warm drift).")
    parser.add_argument("--morrison-sed-cfl-substeps",
                        action=argparse.BooleanOptionalAction, default=True,
                        dest="morrison_sed_cfl_substeps",
                        help="MG2-style CFL sub-stepping of Morrison rain/ice/"
                             "snow/graupel sedimentation (per column nstep = "
                             "1 + floor(max V dt/dz)); default ON (user "
                             "2026-09-22). --no-morrison-sed-cfl-substeps = the "
                             "legacy one-pass form (falls at most one layer per "
                             "call) for reproducing earlier runs.")
    parser.add_argument("--morrison-sed-cfl-substeps-max", type=int,
                        default=ExperimentConfig._field_defaults[
                            "morrison_sed_cfl_substeps_max"],
                        dest="morrison_sed_cfl_substeps_max",
                        help="Static bound of the Morrison CFL sedimentation "
                             "sub-step loop; the cost is LINEAR in it (whole "
                             "Morrison call on 2048 columns, CPU x64: one pass "
                             "10 ms, 16 -> 24 ms, 96 -> 59 ms, 256 -> 131 ms). "
                             "Required counts: 8 on production sigma-36 at "
                             "112.5 s, 90 on CAM L32 at 600 s.")
    parser.add_argument("--morrison-sed-cfl-substeps-strict", action="store_true",
                        default=False, dest="morrison_sed_cfl_substeps_strict",
                        help="With --morrison-sed-cfl-substeps: abort the run "
                             "when any column needs more sub-steps than the "
                             "static cap (otherwise the count is only reported).")
    parser.add_argument("--morrison-do-graupel",
                        action=argparse.BooleanOptionalAction,
                        default=ExperimentConfig._field_defaults[
                            "morrison_do_graupel"],
                        dest="morrison_do_graupel",
                        help="Morrison prognostic graupel category (riming "
                             "onto graupel, frozen rain -> graupel). CAM6's "
                             "MG2 has no graupel; --no-morrison-do-graupel "
                             "routes frozen rain to snow and removes the "
                             "graupel riming sink of cloud water.")
    parser.add_argument("--tropopause-refine", type=float, default=None,
                        dest="tropopause_refine",
                        help="Sigma-coordinate layer redistribution toward "
                             "the tropopause at FIXED nlev (1.0 = uniform, "
                             "bit-identical; 3.0 doubles the levels in "
                             "70-200 hPa, paid for by the mid-troposphere). "
                             "Fixes the unresolved tropical cold point "
                             "without adding levels. Sigma coordinate only.")
    parser.add_argument("--sigma-refine", type=float, default=None, dest="sigma_refine",
                        help="Centre (sigma) of the --tropopause-refine density bump; "
                             "0.12 = tropical cold point (default), ~0.95 = boundary layer.")
    parser.add_argument("--sigma-refine-width", type=float, default=None,
                        dest="sigma_refine_width",
                        help="Log-sigma half-width of the --tropopause-refine bump "
                             "(default 0.45; ~0.06 for a boundary-layer bump).")
    parser.add_argument("--sigma-top", type=float, default=None, dest="sigma_top",
                        help="sigma-lane model lid as a fraction of p_s "
                             "(default 0.01 = 10 hPa)")
    parser.add_argument("--sigma-layout", type=str, default=None, dest="sigma_layout",
                        choices=("standard", "l30_trop_logstrat"),
                        help="sigma-lane level layout: standard (uniform/refined) or "
                             "l30_trop_logstrat (L30 troposphere kept, log-spaced "
                             "stratosphere up to --sigma-top)")
    parser.add_argument("--hard-sat-ice-curve",
                        action=argparse.BooleanOptionalAction, default=False,
                        dest="hard_sat_ice_curve",
                        help="Mixed-phase hard-saturation drain: gate and "
                             "land on the w(T)-blended liquid/ice saturation "
                             "curve below freezing (blended latent heat, "
                             "cold condensate to cloud ice). Fixes the "
                             "TTL ice-supersaturation vapour bias. Requires "
                             "--hard-saturation-adjustment.")
    parser.add_argument("--mpas-ice-skin-prognostic",
                        action=argparse.BooleanOptionalAction, default=False,
                        dest="mpas_ice_skin_prognostic",
                        help="MPAS lane only: prognostic sea-ice skin "
                             "temperature (Semtner 1976 zero-layer conduction "
                             "+ slab thermal inertia) replacing the constant "
                             "T_ice anchor over ice-covered cells — removes "
                             "the year-round 271.35 K pin behind the polar "
                             "tas warm bias. Needs radiation != none.")
    parser.add_argument("--mpas-ice-thickness-m", type=float, default=None,
                        dest="mpas_ice_thickness_m",
                        help="Climatological ice slab thickness [m] for the "
                             "prognostic ice skin (default 2.0; bounds "
                             "[0.1, 10]). Requires --mpas-ice-skin-prognostic.")
    # --cloud-conv-cloud-max closes the AMIP CLI gap for the existing
    # ExperimentConfig.cloud_conv_cloud_max field (--q-c-diagnostic / --rh-crit /
    # --subgrid-autoconv already ship from run_coupled-mirrored #647 + #613).
    parser.add_argument("--cloud-conv-cloud-max", type=float, default=None,
                        dest="conv_cloud_max",
                        help="Cap on convective (Slingo-1987-inspired surrogate) cloud cover "
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
    parser.add_argument("--budget-ledger", action=argparse.BooleanOptionalAction,
                        default=False, dest="budget_ledger",
                        help="Per-process column water/energy budget ledger "
                             "(diagnostics attribution: turbulence/convection/"
                             "microphysics/radiation/other/clips/dynamics). "
                             "Writes segment-mean rates to budget_ledger.npz. "
                             "Single-rank only; default off = byte-identical "
                             "model.")
    parser.add_argument("--budget-ledger-sigma-band", nargs=2, type=float,
                        default=None, metavar=("SIGMA_LO", "SIGMA_HI"),
                        dest="budget_ledger_sigma_band",
                        help="Restrict the budget ledger to a sigma band "
                             "(lo < hi in [0,1], sigma increasing downward). "
                             "The whole-column ledger cannot see a vertical "
                             "REDISTRIBUTION bias -- a mass-flux convection "
                             "scheme's column water row is exactly zero -- so "
                             "a band is what makes it answer which process "
                             "supplies a LAYER. Needs --budget-ledger.")
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


def _resolve_land_model(args, parser=None, argv=None):
    """Apply ``--land-model`` to the two land switches, or report what they mean.

    The AMIP driver has selected its land surface with two independent booleans
    while every other driver has selected it by name.  Four boolean combinations
    encode three states, and the all-false one means NO LAND SURFACE MODEL —
    not a simpler one.  This routes the named choice onto the switches through
    the land package's own vocabulary, so there is one spelling of the concept
    rather than two.

    Passing --land-model together with --slab-land-active or
    --use-multilayer-land is refused outright: silently overriding one with the
    other is how a run ends up not being the run its deck describes.

    Always returns the RESOLVED model name so the caller can print it. What the
    run resolved to is the thing worth logging; the flags are not.
    """
    from legoesm.land.config import describe_land_model, land_model_switches

    # Detect which switches the USER actually typed, not which came out true.
    # Truthiness cannot see an explicit negative: ``--no-use-multilayer-land``
    # sets False, which is also the default, so a truthiness test would let
    # ``--land-model multilayer`` silently overrule a flag the user wrote down.
    # Both spellings of each switch are checked, ``=value`` forms included.
    _switches = ("slab-land-active", "use-multilayer-land")
    _typed = [] if argv is None else [
        f"--{name}" for name in _switches
        if any(a == f"--{name}" or a.startswith(f"--{name}=")
               or a == f"--no-{name}" or a.startswith(f"--no-{name}=")
               for a in argv)]
    # argv unavailable (direct call): fall back to truthiness, which still
    # catches the positive forms.
    explicit = _typed if argv is not None else [
        f for f, v in (("--slab-land-active", args.slab_land_active),
                       ("--use-multilayer-land", args.use_multilayer_land))
        if v]
    if args.land_model is not None:
        # A mask file activates the land tile on its own, so "none" alongside
        # one is a contradiction: the run would have land while its own
        # selector said it did not.  Refuse rather than print a name that is
        # not what the run resolves to.
        if args.land_model == "none" and getattr(args, "land_mask_file", ""):
            msg = ("--land-model none conflicts with --land-mask-file "
                   f"{args.land_mask_file!r}: a land-mask file activates the "
                   "land tile by itself, so the run would have a land surface "
                   "while asking for none. Drop one of the two.")
            if parser is not None:
                parser.error(msg)
            raise SystemExit(msg)
        if explicit:
            msg = (f"--land-model {args.land_model} conflicts with "
                   f"{' and '.join(explicit)}; pass either the named model or "
                   "the individual switches, not both.")
            if parser is not None:
                parser.error(msg)
            raise SystemExit(msg)
        for field, value in land_model_switches(args.land_model).items():
            setattr(args, field, value)

    return describe_land_model(
        slab_land_active=args.slab_land_active,
        use_multilayer_land=args.use_multilayer_land,
        has_land_mask=bool(getattr(args, "land_mask_file", "")))


def _default_discretization_for_grid(grid_type: str) -> str:
    """The dycore discretization to use when the user gave a grid but no
    ``--discretization``.  The SCVT Voronoi mesh supports only 'mpas'; every
    other grid defaults to 'centered'.  Shared by ``_postprocess_args`` and
    ``build_config_from_args`` so a caller that skips postprocess still resolves
    a supported (model_type, discretization, grid_type) triple (codex r2)."""
    if grid_type in ("voronoi", "mpas", "mpas_voronoi", "icosahedral"):
        return "mpas"
    return "centered"


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    # Defensive boundary: --grid-type/--discretization carry a None sentinel
    # default (explicitness tracking for the --truncation conflict guard in
    # _postprocess_args). A caller that skips _postprocess_args must still get
    # the production defaults, not None (codex review 2026-07-17).
    if args.grid_type is None:
        args.grid_type = "cubed_sphere"
    if args.discretization is None:
        args.discretization = _default_discretization_for_grid(args.grid_type)
    grid_config = GridConfig(
        grid_type=args.grid_type,
        resolution=args.resolution,
        nlev=args.nlev,
        vertical_coord=args.vertical_coord,
        p_top_Pa=args.p_top if args.p_top is not None else 200.0,
        stretching=args.stretching if args.stretching is not None else 2.0,
        transition_exponent=(args.transition_exponent
                             if args.transition_exponent is not None else 3),
        tropopause_refine=(args.tropopause_refine
                           if args.tropopause_refine is not None else 1.0),
        sigma_top=(args.sigma_top if args.sigma_top is not None else 0.01),
        sigma_refine=(args.sigma_refine if args.sigma_refine is not None else 0.12),
        sigma_refine_width=(args.sigma_refine_width
                            if args.sigma_refine_width is not None else 0.45),
        sigma_layout=(args.sigma_layout if args.sigma_layout is not None else "standard"),
        use_duogrid=getattr(args, "use_duogrid", False),
    )

    dycore_config = DycoreConfig(
        discretization=args.discretization,
        dt=args.dt,
        fv3_duo_windows=args.fv3_duo_windows,
        fv3_duo_window_pad=args.fv3_duo_window_pad,
        fv3_duo_column_lane=args.fv3_duo_column_lane,
        hyperdiff_scale=args.hyperdiff_scale,
        corner_fill=args.corner_fill,
        a_h_scale=args.a_h_scale,
        k_h_scale=args.k_h_scale,
        div_damp_scale=args.div_damp_scale,
        moisture_flux_form=args.moisture_flux_form,
        mpas_nu_vert4_T=args.mpas_nu_vert4_t,
        mpas_conservative_tracer_clamp=args.mpas_conservative_tracer_clamp,
        mpas_vert_advection_scheme=args.mpas_vert_advection_scheme,
        mpas_sponge_del2_top_layers=(args.mpas_sponge_del2_top_layers
                                     if args.mpas_sponge_del2_top_layers is not None else 0),
        mpas_sponge_del2_top_factor=(args.mpas_sponge_del2_top_factor
                                     if args.mpas_sponge_del2_top_factor is not None else 1.0),
        mpas_div_damp4_scale=(args.mpas_div_damp4_scale
                              if args.mpas_div_damp4_scale is not None else 0.0),
        conservation_fixer=args.conservation_fixer,
        fix_mass=args.fix_mass,
        implicit_grav_wave_use_pcg=args.implicit_grav_wave_use_pcg,
        implicit_grav_wave_damping=args.implicit_grav_wave_damping,
        # #1028: cube winds stay D-staggered between steps.
        persistent_dgrid=args.persistent_dgrid,
        # Stage 3-E: polar filter for lat-lon C-grid pole-CFL relief.
        use_polar_filter=args.use_polar_filter,
        polar_filter_cutoff_deg=args.polar_filter_cutoff_deg,
        polar_filter_max_wave_speed=args.polar_filter_max_wave_speed,
        # #836 top sponge (default OFF -> bit-identical dycore).
        sponge_coeff=args.sponge_coeff,
        sponge_width_m=args.sponge_width_m,
        sponge_shape=args.sponge_shape,
        sponge_scale_height_m=args.sponge_scale_height_m,
        # #1029 ω-side SB81 conversion (default OFF -> bit-identical).
        sb81_omega_conversion=args.sb81_omega_conversion,
        # Task #25: time integrator selection.
        time_integrator=args.time_integrator,
    )

    output_config = OutputConfig(
        output_dir=args.output or "",
        diag_days=args.diag_days,
        checkpoint_days=args.checkpoint_days,
        cmip_resolution_deg=args.cmip_resolution_deg,
        max_wallclock_seconds=args.max_wallclock_seconds,
        monthly_means=args.monthly_means,
        cmip_output=args.cmip_output,
        clear_sky_diag=args.clear_sky_diag,
        budget_ledger=args.budget_ledger,
        budget_ledger_sigma_band=(tuple(args.budget_ledger_sigma_band)
                                  if args.budget_ledger_sigma_band else None),
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
        physics_update_steps=args.physics_update_steps,
        cld_macmic_num_steps=args.cld_macmic_num_steps,
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
        use_clubb_cloud_fraction=args.use_clubb_cloud_fraction,
        clubb_prognostic=args.clubb_prognostic,
        clubb_liquid_partition=args.clubb_liquid_partition,
        clubb_trop_cloud_top_press=args.clubb_trop_cloud_top_press,
        **{_f: getattr(args, _f)
           for _f in {**ZM_SCALAR_FIELDS, **CLUBB_SCALAR_FIELDS}},
        clubb_q_flux_scale=args.clubb_q_flux_scale,
        clubb_q_flux_scale_sigma_band=(tuple(args.clubb_q_flux_scale_sigma_band)
                                       if args.clubb_q_flux_scale_sigma_band
                                       is not None else None),
        microphysics=args.microphysics,
        nc_from_aerosol=args.aerosol_ccn,
        subgrid_autoconversion=args.subgrid_autoconversion,
        hard_saturation_adjustment=args.hard_saturation_adjustment,
        hard_sat_adjust_threshold=args.hard_sat_adjust_threshold,
        hard_sat_max_heating_K=args.hard_sat_max_heating_K,
        convective_precip_efficiency=args.convective_precip_efficiency,
        convective_precip_split=args.convective_precip_split,
        convective_rain_to_surface=args.convective_rain_to_surface,
        autoconv_q_c_crit=args.autoconv_q_c_crit,
        autoconv_pe_max=args.autoconv_pe_max,
        convective_buoyancy_death_memory=args.convective_buoyancy_death_memory,
        convection=args.convection,
        turbulence=args.turbulence,
        gravity_wave_drag=args.gravity_wave_drag,
        # Tuned air-sea + cloud calibration (mirror run_coupled).
        surface_bulk_scheme=args.surface_bulk_scheme,
        surface_stability_scheme=args.surface_stability_scheme,
        surface_gustiness_zi=args.surface_gustiness_zi,
        surface_z_ref_model_level=args.surface_z_ref_model_level,
        surface_ocean_q_sfc_saline=args.surface_ocean_q_sfc_saline,
        hb_kvf_min=args.hb_kvf_min,
        louis_cloudtop_entrainment_efficiency=args.louis_cloudtop_entrainment_efficiency,
        surface_thermo_convention=args.bulk_thermo_convention,
        cloud_q_c_diagnostic=args.cloud_q_c_diagnostic,
        cloud_rh_crit=args.cloud_rh_crit,
        cloud_clubb_cf_override_strength=args.cloud_clubb_cf_override_strength,
        cloud_clubb_cf_override_floor=args.cloud_clubb_cf_override_floor,
        cloud_inhomogeneity_factor=args.cloud_inhomogeneity_factor,
        cloud_optics_inhomogeneity=args.cloud_optics_inhomogeneity,
        cloud_partial_coverage_optics=args.cloud_partial_coverage_optics,
        cloud_saturation_scheme=args.cloud_saturation_scheme,
        cloud_vertical_overlap_optics=args.cloud_vertical_overlap_optics,
        cloud_n_subcolumns=args.cloud_n_subcolumns,
        cloud_fsd=args.cloud_fsd,
        cloud_p_xr=args.cloud_p_xr,
        cloud_alpha_xr=args.cloud_alpha_xr,
        cloud_cover_condensate_q_ref=args.cloud_cover_condensate_q_ref,
        cloud_cam6_rhmini=args.cloud_cam6_rhmini,
        cloud_cam6_rhmaxi=args.cloud_cam6_rhmaxi,
        cloud_cam6_rhminis=args.cloud_cam6_rhminis,
        cloud_cam6_rhmaxis=args.cloud_cam6_rhmaxis,
        cloud_cap_floor_on=args.cloud_cap_floor_on,
        cloud_cap_floor_lat_deg=args.cloud_cap_floor_lat_deg,
        cloud_cap_floor_p_max_pa=args.cloud_cap_floor_p_max_pa,
        cloud_cap_floor_cf=args.cloud_cap_floor_cf,
        cloud_cap_floor_q_c=args.cloud_cap_floor_q_c,
        snow_age_activation_K=args.snow_age_activation_K,
        land_snow_tau_days=args.land_snow_tau_days,
        land_soil_freeze_thaw=args.land_soil_freeze_thaw,
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
        land_update_seconds=args.land_update_seconds,
        multilayer_n_layers=args.multilayer_n_layers,
        multilayer_soil_depth=args.multilayer_soil_depth,
        land_calibrated_physics=args.land_calibrated_physics,
        clm_surfdata_path=args.clm_surfdata_path,
        transient_land_cover=args.transient_land_cover,
        land_cover_surfdata=args.land_cover_surfdata,
        albedo_land_path=args.albedo_land_file,
        albedo_land_month=args.albedo_land_month,
        subgrid_orography_path=args.subgrid_orography_file,
        slab_land_active=args.slab_land_active,
        land_interface_flux=args.land_interface_flux,
        C_land=args.C_land,
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
        land_soil_init=args.land_soil_init,
        land_surface_scheme=args.land_surface_scheme,
        clm_ml_use_surfdata_pft=args.clm_ml_use_surfdata_pft,
        land_ic_path=args.land_ic,
        sponge_enabled=args.sponge_enabled,
        sponge_coeff_per_day=(args.sponge_coeff_per_day
                              if args.sponge_coeff_per_day is not None
                              else _EXPERIMENT_DEFAULTS.sponge_coeff_per_day),
        sponge_sigma_top=(args.sponge_sigma_top
                          if args.sponge_sigma_top is not None
                          else _EXPERIMENT_DEFAULTS.sponge_sigma_top),
        mpas_land_lapse_K_per_km=(
            args.mpas_land_lapse_K_per_km
            if args.mpas_land_lapse_K_per_km is not None
            else _EXPERIMENT_DEFAULTS.mpas_land_lapse_K_per_km),
        mpas_land_beta=(args.mpas_land_beta
                        if args.mpas_land_beta is not None
                        else _EXPERIMENT_DEFAULTS.mpas_land_beta),
        mpas_land_beta_soil=args.mpas_land_beta_soil,
        mpas_land_stress_from_land=args.mpas_land_stress_from_land,
        mpas_land_params_refresh=args.mpas_land_params_refresh,
        mpas_qv_smooth_del2_m2s=(
            args.mpas_qv_smooth_del2_m2s
            if args.mpas_qv_smooth_del2_m2s is not None
            else _EXPERIMENT_DEFAULTS.mpas_qv_smooth_del2_m2s),
        mpas_qv_smooth_del4_m4s=(
            args.mpas_qv_smooth_del4_m4s
            if args.mpas_qv_smooth_del4_m4s is not None
            else _EXPERIMENT_DEFAULTS.mpas_qv_smooth_del4_m4s),
        hard_sat_ice_curve=args.hard_sat_ice_curve,
        homogeneous_ice_nucleation=args.homogeneous_ice_nucleation,
        morrison_flavor=args.morrison_flavor,
        morrison_warm_rain_scheme=args.morrison_warm_rain_scheme,
        morrison_autocon_fact=args.morrison_autocon_fact,
        morrison_accre_enhan_fact=args.morrison_accre_enhan_fact,
        morrison_sed_cfl_substeps=args.morrison_sed_cfl_substeps,
        morrison_sed_cfl_substeps_max=args.morrison_sed_cfl_substeps_max,
        morrison_sed_cfl_substeps_strict=args.morrison_sed_cfl_substeps_strict,
        morrison_do_graupel=args.morrison_do_graupel,
        morrison_warm_rain_incloud=args.morrison_warm_rain_incloud,
        hines_total_rms_wind=(
            args.hines_total_rms_wind
            if args.hines_total_rms_wind is not None
            else _EXPERIMENT_DEFAULTS.hines_total_rms_wind),
        hines_launch_p=(
            args.hines_launch_p
            if args.hines_launch_p is not None else 0.0),
        hines_Fmax=(args.hines_Fmax if args.hines_Fmax is not None
                    else _EXPERIMENT_DEFAULTS.hines_Fmax),
        e3sm_cam_source=(args.e3sm_cam_source
                         if args.e3sm_cam_source is not None
                         else _EXPERIMENT_DEFAULTS.e3sm_cam_source),
        e3sm_cam_pgwv=(args.e3sm_cam_pgwv
                       if args.e3sm_cam_pgwv is not None
                       else _EXPERIMENT_DEFAULTS.e3sm_cam_pgwv),
        e3sm_cam_effgw=(args.e3sm_cam_effgw
                        if args.e3sm_cam_effgw is not None
                        else _EXPERIMENT_DEFAULTS.e3sm_cam_effgw),
        e3sm_cam_taubgnd=(args.e3sm_cam_taubgnd
                          if args.e3sm_cam_taubgnd is not None
                          else _EXPERIMENT_DEFAULTS.e3sm_cam_taubgnd),
        e3sm_cam_c0=(args.e3sm_cam_c0
                     if args.e3sm_cam_c0 is not None
                     else _EXPERIMENT_DEFAULTS.e3sm_cam_c0),
        e3sm_cam_launch_p=(args.e3sm_cam_launch_p
                           if args.e3sm_cam_launch_p is not None
                           else _EXPERIMENT_DEFAULTS.e3sm_cam_launch_p),
        e3sm_cam_latitude_taper=(
            args.e3sm_cam_latitude_taper
            if args.e3sm_cam_latitude_taper is not None
            else _EXPERIMENT_DEFAULTS.e3sm_cam_latitude_taper),
        e3sm_cam_effgw_cm=(args.e3sm_cam_effgw_cm
                           if args.e3sm_cam_effgw_cm is not None
                           else _EXPERIMENT_DEFAULTS.e3sm_cam_effgw_cm),
        e3sm_cam_effgw_beres=(args.e3sm_cam_effgw_beres
                              if args.e3sm_cam_effgw_beres is not None
                              else _EXPERIMENT_DEFAULTS.e3sm_cam_effgw_beres),
        e3sm_cam_frontgfc=(args.e3sm_cam_frontgfc
                           if args.e3sm_cam_frontgfc is not None
                           else _EXPERIMENT_DEFAULTS.e3sm_cam_frontgfc),
        e3sm_cam_beres_variant=(
            args.e3sm_cam_beres_variant
            if args.e3sm_cam_beres_variant is not None
            else _EXPERIMENT_DEFAULTS.e3sm_cam_beres_variant),
        e3sm_cam_mfcc_table_path=(
            args.e3sm_cam_mfcc_table_path
            if args.e3sm_cam_mfcc_table_path is not None
            else _EXPERIMENT_DEFAULTS.e3sm_cam_mfcc_table_path),
        e3sm_cam_dttke_intrinsic=(
            args.e3sm_cam_dttke_intrinsic
            if args.e3sm_cam_dttke_intrinsic is not None
            else _EXPERIMENT_DEFAULTS.e3sm_cam_dttke_intrinsic),
        mcfarlane_tau_max=(
            args.mcfarlane_tau_max if args.mcfarlane_tau_max is not None
            else _EXPERIMENT_DEFAULTS.mcfarlane_tau_max),
        mcfarlane_k_wave=(
            args.mcfarlane_k_wave if args.mcfarlane_k_wave is not None
            else _EXPERIMENT_DEFAULTS.mcfarlane_k_wave),
        mpas_ice_skin_prognostic=args.mpas_ice_skin_prognostic,
        mpas_ice_thickness_m=(
            args.mpas_ice_thickness_m
            if args.mpas_ice_thickness_m is not None
            else _EXPERIMENT_DEFAULTS.mpas_ice_thickness_m),
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
        bechtold_subsidence_solve=args.bechtold_subsidence_solve,
        bechtold_rain_vapor_sink=args.bechtold_rain_vapor_sink,
        zm_land_fraction=args.zm_land_fraction,
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
        bechtold_enable_cmt=args.bechtold_enable_cmt,
        bechtold_M_b_max=args.bechtold_M_b_max,
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


_SPECTRAL_PROGNOSTIC_CONVECTION = frozenset({
    "tiedtke", "bechtold", "zhang_mcfarlane", "kain_fritsch", "mass_flux",
    "edmf", "emanuel"})


def _apply_spectral_scheme_fallback(args: argparse.Namespace, argv,
                                    parser: argparse.ArgumentParser | None = None
                                    ) -> argparse.Namespace:
    """Spectral/gaussian AMIP: the spectral run loop cannot thread a prognostic
    physics carry yet (issue #405), so prognostic convection / gravity-wave-drag
    are refused deep in setup. When the user did NOT explicitly pick them, fall
    back to the diagnostic schemes the spectral loop CAN run — convection ->
    ``sbm``, GWD -> ``rayleigh`` — with a notice, so gaussian AMIP runs out of the
    box. Explicit ``--convection`` / ``--gravity-wave-drag`` are honoured verbatim
    (and still correctly refused by the loop if prognostic — the user's call).
    The faithful prognostic-on-spectral path is threading PhysicsState through the
    spectral step (the #405 follow-up), NOT this scheme swap. No-op off spectral."""
    if getattr(args, "discretization", None) != "spectral":
        return args
    # #405: the spectral run loop now THREADS the prognostic PhysicsState carry
    # on the LEAPFROG path (single physics eval per step), so a prognostic
    # scheme runs FAITHFULLY there — do not downgrade it.  The downgrade below
    # applies only to the per-RK-stage ssp_rk3 path, where a single-step carry
    # is ill-defined.  (Auto-selecting leapfrog_si + its semi-implicit matrices
    # for an `auto` integrator is a further usability step; today the prognostic
    # spectral path is reached via an explicit `--time-integrator leapfrog_si`.)
    _integ = str(getattr(args, "time_integrator", "auto")).lower()
    if _integ in ("leapfrog", "leapfrog_si"):
        print("[run_amip] spectral leapfrog path: prognostic physics carry is "
              "threaded (#405) — schemes run faithfully (no diagnostic swap).",
              flush=True)
        return args
    toks = list(argv or [])
    has = lambda f: any(a == f or a.startswith(f + "=") for a in toks)
    if not has("--convection") and args.convection in _SPECTRAL_PROGNOSTIC_CONVECTION:
        # Reject a silent-no-op: the user did not pick --convection (so we
        # downgrade to diagnostic sbm), but they DID pass an option that only
        # the mass-flux convection honors — after the swap it would be
        # silently ignored (codex r2).  Make them choose explicitly.
        _convection_dependent_flags = [
            f for f in ("--convective-precip-efficiency",
                        "--convective-precip-split", "--autoconv-q-c-crit",
                        "--autoconv-pe-max", "--convective-buoyancy-death-memory")
            if has(f)
        ]
        if _convection_dependent_flags:
            _msg = (
                f"the spectral/gaussian loop cannot thread prognostic "
                f"convection '{args.convection}' yet (issue #405) and would "
                f"downgrade to diagnostic 'sbm', but you passed "
                f"{_convection_dependent_flags}, which only a mass-flux "
                f"convection honors — it would be silently ignored. Either "
                f"drop those flags or pass an explicit --convection that "
                f"supports them (and a --time-integrator leapfrog_si spectral "
                f"path that threads the prognostic carry)."
            )
            if parser is not None:
                parser.error(_msg)
            raise SystemExit(f"run_amip: error: {_msg}")
        print(f"[run_amip] spectral loop cannot thread prognostic convection "
              f"'{args.convection}' yet (issue #405) -> diagnostic 'sbm'. "
              f"Pass --convection to override.", flush=True)
        args.convection = "sbm"
    if not has("--gravity-wave-drag") and "mcfarlane" in str(args.gravity_wave_drag):
        print(f"[run_amip] spectral loop cannot thread prognostic GWD "
              f"'{args.gravity_wave_drag}' yet (issue #405) -> diagnostic "
              f"'rayleigh'. Pass --gravity-wave-drag to override.", flush=True)
        args.gravity_wave_drag = "rayleigh"
    return args


def _postprocess_args(args: argparse.Namespace, parser: argparse.ArgumentParser,
                      argv: list[str] | None = None) -> argparse.Namespace:
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

    # SST/SIC forcing is NOT stored in the checkpoint, and driver.setup()
    # (which loads forcing) runs before load_checkpoint — so a restart of a
    # real-data run still needs --forcing-path.  The old message exempted
    # --restart-from, which was false: it deferred the failure to a confusing
    # deep "AMIPForcingConfig.path is empty" inside setup (audit 2026-07-17).
    if args.forcing_path is None and args.dataset != "analytical":
        parser.error(
            "--forcing-path required for --dataset "
            f"{args.dataset!r} (unless --dataset analytical). Forcing is not "
            "stored in the checkpoint, so a --restart-from run must re-supply "
            "the same --forcing-path."
        )
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
    # MPAS implements the zenith ocean curve in its own daily surface-albedo
    # assembly, so only the SPECTRAL standalone path still has nowhere to put
    # it.  Narrowed rather than deleted: a flag that is silently ignored is the
    # defect this guard exists to prevent.
    if args.dynamic_albedo and args.discretization == "spectral":
        parser.error("--dynamic-albedo has no effect on the SPECTRAL "
                     "standalone radiation path, which uses the RRTMGPConfig "
                     "constant surface albedo and would silently ignore the "
                     "flag. It is supported on cubed_sphere/latlon (coupled "
                     "physics pipeline) and on MPAS (daily tile-blended "
                     "surface albedo).")
    if args.use_multilayer_land and args.discretization == "spectral":
        parser.error("--use-multilayer-land runs inside the coupled physics "
                     "pipeline or the MPAS driver loop; the SPECTRAL "
                     "standalone physics carries a PASSIVE land tile and "
                     "cannot step the soil column. Pass "
                     "--no-use-multilayer-land to override a --config YAML "
                     "that enables it.")
    # MPAS port (tasks/mpas_land_port.md): the multilayer tile is stepped in
    # the MPAS driver loop (explicit flux coupling via forcing['T_sfc']).
    # clm_ml still needs the coupled pipeline's per-column canopy grid
    # threading, which this lane does not have.  simple_seb and two_leaf are
    # both dispatched by the MPAS land step, so only clm_ml is refused here.
    if (args.use_multilayer_land
            and args.land_surface_scheme == "clm_ml"
            and (args.grid_type in ("voronoi", "icosahedral", "mpas_voronoi",
                                    "mpas")
                 or args.discretization == "mpas")):
        parser.error(
            "--land-surface-scheme clm_ml is not wired on the MPAS lane "
            "(it needs the coupled pipeline's per-column canopy grid "
            "threading); use two_leaf or simple_seb with "
            "--use-multilayer-land on MPAS.")
    # two_leaf IS wired on MPAS: the land step dispatches to it, and its
    # solved canopy-air humidity now reaches the turbulence through the traced
    # beta channel (the guard here used to refuse it alongside clm_ml, which
    # made a resistance-based land surface unreachable on this lane).
    # Canopy surface schemes run INSIDE the multilayer land tile; without
    # --use-multilayer-land the slab land runs and the scheme is silently dropped
    # (the user asked for a canopy, got the slab).  Fail early rather than degrade
    # silently.
    #
    # Only when the canopy was asked for EXPLICITLY.  The two-leaf canopy is now
    # the DEFAULT surface scheme, so gating on the value alone would refuse every
    # slab-only run that never mentioned a canopy at all.  A default that cannot
    # be left alone is not a default.
    # "Asked for" means the COMMAND LINE or a --config YAML.  A YAML key sets
    # argparse DEFAULTS, so testing argv alone silently rewrote a config that
    # explicitly requested a canopy — the exact silent degradation this guard
    # exists to prevent (codex).  ``_config_keys`` is stamped by the loader.
    _argv = argv if argv is not None else sys.argv[1:]
    _canopy_explicit = (
        any(a == "--land-surface-scheme" or a.startswith("--land-surface-scheme=")
            for a in _argv)
        or "land_surface_scheme" in getattr(args, "_config_keys", ()))
    if (args.land_surface_scheme in ("two_leaf", "clm_ml")
            and not args.use_multilayer_land):
        if _canopy_explicit:
            parser.error(
                f"--land-surface-scheme {args.land_surface_scheme} is a canopy "
                "scheme that runs inside the multilayer land tile and has NO "
                "effect on the slab land — it would be silently dropped. Pass "
                "--use-multilayer-land, or use --land-surface-scheme simple_seb.")
        # Not asked for: fall back to the slab's own surface treatment, and say so
        # UNCONDITIONALLY, so a slab run never silently claims a canopy it does
        # not have.  A comment promising to say so is not saying so (codex).
        print(
            f"WARNING: land surface scheme resolved to 'simple_seb'. The default "
            f"'{args.land_surface_scheme}' is a canopy that runs inside the "
            "multilayer land tile, and this run has no such tile (no "
            "--use-multilayer-land), so the slab's own surface treatment is used "
            "instead. simple_seb is an ACADEMIC scheme: its evaporation runs at "
            "potential with stomata off and self-extinguishes with them on. Pass "
            "--use-multilayer-land for the canopy.", file=sys.stderr)
        # DELIBERATELY not rewritten.  The field is documented as affecting
        # multilayer runs only, so it is inert here by construction, and the
        # warning above is what stops that being silent.  Assigning a value
        # instead would make the field NON-DEFAULT on every slab run, which the
        # strict single-purpose lanes reject outright (fv3_duo refuses any
        # non-default field it does not consume).
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
        # Per-grid default: the SCVT Voronoi mesh has exactly one dycore
        # discretization ('mpas'); the previous unconditional 'centered'
        # default made bare ``--grid-type voronoi`` die at the dycore
        # factory with an unsupported (hydrostatic, centered, mpas) triple
        # (2026-07-21 audit — cross-grid smoke).
        args.discretization = _default_discretization_for_grid(args.grid_type)

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
    if getattr(args, "cloud_cap_floor_on", False):
        raise SystemExit("--cloud-cap-floor cannot be combined with --aimip-classical-checkpoint: "
                         "the trained cloud config is prebuilt and would not carry the floor")
    from legoesm.ml.checkpoint_io import load_checkpoint_or_fail
    from legoesm.training.aimip_params import AIMIPClassicalParams
    _p = load_checkpoint_or_fail(
        args.aimip_classical_checkpoint, AIMIPClassicalParams.from_defaults(),
        what="--aimip-classical-checkpoint")
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
    ``gustiness_w_zi`` + ``stability_scheme`` and the two run-resolved surface
    switches (``z_ref_model_level``, ``ocean_q_sfc_saline``) onto the trained
    Louis config, keeping the trained ``Cd_neutral``/``Ch_neutral``/``z0``
    (the MOST/COARE schemes ignore the neutral ``Cd``/``Ch`` but DO use
    ``z0``, so preserving all three is correct).  The switches were dropped
    here before, so ``--surface-z-ref-model-level`` never reached the trained
    lane.  No-op when there is no prior turbulence config / surface.
    """
    prev_surf = getattr(prev_turb_config, "surface", None)
    if prev_surf is None or getattr(louis_config, "surface", None) is None:
        return louis_config
    return louis_config._replace(
        surface=louis_config.surface._replace(
            bulk_scheme=prev_surf.bulk_scheme,
            gustiness_w_zi=prev_surf.gustiness_w_zi,
            stability_scheme=prev_surf.stability_scheme,
            z_ref_model_level=prev_surf.z_ref_model_level,
            ocean_q_sfc_saline=prev_surf.ocean_q_sfc_saline))


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
        _cfg_keys = load_yaml_config(
            pre.config, parser,
            example_keys="'convection', 'microphysics', 'surface_bulk_scheme', "
                         "'q_c_diagnostic', 'gustiness_zi', "
                         "'bulk_thermo_convention', 'convective_cloud'")
        parser.set_defaults(**_cfg_keys)
        # Remember WHICH keys the file set.  A YAML key arrives as an argparse
        # DEFAULT, so downstream checks cannot otherwise tell "the user asked for
        # this" from "nobody mentioned it" — and one such check was silently
        # rewriting an explicitly requested canopy scheme (codex).
        parser.set_defaults(_config_keys=frozenset(_cfg_keys))

    args = parser.parse_args(argv)
    # Land surface: map --land-model onto the two switches (and refuse the
    # contradictory combination) BEFORE anything reads them.  The returned name
    # is what the run actually resolved to, which is the thing worth logging —
    # "none" here means there is no land surface model at all, so it is printed
    # whether or not the flag was used.
    # ``main(argv=None)`` means "read sys.argv", so resolve it here — passing
    # the bare None would drop the resolver back to truthiness on the ordinary
    # command-line path, i.e. exactly where the explicit-negative check matters.
    _land_model = _resolve_land_model(
        args, parser, argv if argv is not None else sys.argv[1:])
    print(f"[run_amip] Land surface model: {_land_model} "
          f"(slab_land_active={args.slab_land_active}, "
          f"use_multilayer_land={args.use_multilayer_land}, "
          f"land_mask_file={getattr(args, 'land_mask_file', '') or '<none>'})")
    # Postprocess FIRST: it resolves the grid/discretization sentinels
    # (``--truncation 21`` alone sets discretization="spectral" only there),
    # and the spectral fallback keys off ``args.discretization == "spectral"``
    # — calling it on the unresolved sentinel made it a silent no-op for the
    # ``--truncation``-only spelling, so gaussian AMIP died at setup on the
    # prognostic default schemes (2026-07-21 audit — cross-grid smoke).
    args = _postprocess_args(args, parser, argv if argv is not None else sys.argv[1:])

    _apply_spectral_scheme_fallback(
        args, argv if argv is not None else sys.argv[1:], parser)

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

    # The strict sedimentation-overflow abort is an equinox ``error_if``, i.e.
    # a host callback: it needs a CPU device to place its inputs on, which the
    # GPU lane (``JAX_PLATFORMS=cuda``) does not have.  Refuse at startup
    # instead of dying mid-run (the first CAM6 60-day arm died at day 2).
    # AFTER the multicontroller init: querying devices initializes the backend,
    # and jax.distributed.initialize must run before any backend work, so the
    # guard used to break a valid --multicontroller launch (codex 2026-09-22).
    if getattr(args, "morrison_sed_cfl_substeps_strict", False):
        from legoesm.driver.model_driver import (
            require_cpu_for_strict_sedimentation,
        )
        require_cpu_for_strict_sedimentation()

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
    # Apply the --params calibration layer (issue #691) in two stages:
    #   (1) parameters with a verified flat ExperimentConfig scalar go through
    #       the scalar map PRE-setup (the pipeline builds their scheme configs
    #       from those scalars);
    #   (2) every other atm parameter routes into the BUILT pipeline's scheme
    #       *Config NamedTuples POST-setup via apply_params_to_pipeline — the
    #       CLUBB-style path that makes all spec'd closure constants settable
    #       without a per-parameter scalar.  Deferred here, applied after
    #       driver.setup() below.
    _pipeline_params: dict = {}
    if getattr(args, "params", None):
        from legoesm.driver.run_config_yaml import (
            apply_params_to_config,
            build_atm_scalar_param_map,
            load_params_config,
        )
        # #1509: capture what --params actually applied. Class-routed
        # values land on nested scheme configs that resolved_config does
        # not reach, so without this the manifest records only the params
        # FILE PATH and a reader months later cannot tell which values
        # produced the run. Defined unconditionally so the driver
        # threading below is well-defined even when nothing is mapped.
        _params_applied: dict = {}
        _all_params = load_params_config(args.params)
        _scalar_map = build_atm_scalar_param_map()
        _mapped = {q: v for q, v in _all_params.items() if q in _scalar_map}
        _pipeline_params = {
            q: v for q, v in _all_params.items() if q not in _scalar_map}
        if _mapped:
            config = apply_params_to_config(
                config, _mapped, driver="run_amip",
                scalar_param_map=_scalar_map,
                record=_params_applied)

    from legoesm.driver.model_driver import ModelDriver

    driver = ModelDriver(config)
    # Threaded onto the driver rather than through its constructor so no other
    # caller's signature changes; the manifest writer reads it if present.
    if "_params_applied" in dir():
        driver._params_applied = _params_applied
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

    # Stage (2) of --params: class-routed overrides into the built pipeline's
    # scheme configs.  Applied LAST — after the AIMIP-trained injection and the
    # Sundqvist CLI overrides above — so an explicit calibration file wins over
    # every other source (same precedence as the scalar-mapped stage, which
    # wins over the ExperimentConfig defaults it replaces).
    if _pipeline_params:
        # LANE GUARD (mirrors the use_clubb_cloud_fraction guard in
        # model_driver.run): the MPAS / spectral / lat-lon-SPMD / tiled-cube
        # rollouts REBUILD their scheme configs from the flat ExperimentConfig
        # (convection_config_for(cfg) etc.), not from driver.physics, so a
        # post-setup override would be silently ignored there while this
        # message claimed it was routed (codex review 2026-08-02). Refuse
        # loudly; the scalar-mapped stage (1) still works in every lane.
        _grid_t = config.grid.grid_type
        _disc = config.dycore.discretization
        _latlon_spmd = bool(getattr(config, "enable_latlon_spmd", False))
        _devcfg = getattr(driver, "_device_config", None)
        _tiled_cube = (
            _grid_t == "cubed_sphere" and _devcfg is not None
            and getattr(_devcfg, "mesh", None) is not None
            and tuple(getattr(_devcfg, "tiling", (1, 1))) != (1, 1))
        if _grid_t == "mpas" or _disc == "spectral" \
                or _latlon_spmd or _tiled_cube:
            raise SystemExit(
                "--params: scheme-config parameter(s) "
                f"{sorted(_pipeline_params)} route into the built pipeline's "
                "config attributes, which the MPAS / spectral / lat-lon-SPMD "
                "/ tiled-cube rollouts do not consume (they rebuild configs "
                "from the flat ExperimentConfig). Got "
                f"grid={_grid_t!r}, discretization={_disc!r}, "
                f"latlon_spmd={_latlon_spmd}, tiled_cube={_tiled_cube}. "
                "Use scalar-mapped parameters (build_atm_scalar_param_map) "
                "in these lanes, or run the single-device per-step lane."
            )
        from legoesm.driver.run_config_yaml import apply_params_to_pipeline
        _applied = apply_params_to_pipeline(
            driver.physics, _pipeline_params, driver="run_amip")
        if _is_root:
            print(f"--params: routed {len(_applied)} scheme-config "
                  f"parameter(s) into the built pipeline: "
                  f"{', '.join(sorted(_applied))}")

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
