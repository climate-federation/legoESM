#!/usr/bin/env python
"""Run a fully coupled Earth System Model simulation.

Supports multiple configuration presets:
  aquaplanet    — Slab ocean everywhere, no land, no carbon
  slab_simple   — Slab ocean + slab bucket land
  slab_pft      — Slab ocean + PFT-weighted land parameters
  slab_richards — Slab ocean + Richards' equation soil hydrology
  slab_carbon   — + DifferLand carbon + atmospheric CO2 tracer
  full_coupled  — + ocean biogeochemistry CO2

Example usage::

    JAX_ENABLE_X64=1 python scripts/run_coupled.py \\
        --preset aquaplanet --days 365 --resolution 16

    JAX_ENABLE_X64=1 python scripts/run_coupled.py \\
        --preset slab_carbon --days 730 --resolution 16 --nlev 20
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path for test_cases imports
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import jax

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_coupled")


def main():
    parser = argparse.ArgumentParser(
        description="Run a fully coupled ESM simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Preset
    parser.add_argument(
        "--preset", default="aquaplanet",
        choices=["aquaplanet", "slab_simple", "slab_pft",
                 "slab_richards", "slab_carbon", "full_coupled"],
        help="Configuration preset (default: aquaplanet)",
    )

    # Atmosphere
    parser.add_argument("--resolution", "-n", type=int, default=16,
                        help="Cubed-sphere resolution C{N} (default: 16)")
    parser.add_argument("--nlev", type=int, default=20,
                        help="Number of vertical levels (default: 20)")
    parser.add_argument("--dt", type=float, default=450.0,
                        help="Atmosphere time step [s] (default: 450)")
    parser.add_argument("--days", type=int, default=30,
                        help="Simulation duration [days] (default: 30)")
    parser.add_argument("--radiation", default="rrtmgp",
                        choices=["gray", "rrtmg", "rrtmgp"],
                        help="Radiation scheme (default: rrtmgp — true CMIP6 "
                             "GHG/cloud-radiative transfer; use --minimal-physics "
                             "or --radiation gray for a cheap idealized run)")
    parser.add_argument(
        "--rad-update-steps", type=int, default=4,
        help="Call radiation every N physics steps (Issue #316: N>1 dispatches "
             "the other steps to a no-radiation segment variant, cutting the "
             "rrtmgp compiled-segment compile from O(hours) to O(minutes); "
             "physically fine since radiation evolves slowly). Default 4 "
             "(amortizes the rrtmgp default).",
    )
    parser.add_argument(
        "--unfused-radiation", action=argparse.BooleanOptionalAction, default=True,
        help="Run radiation as a SEPARATE host-level jit (not fused into the "
             "lax.scan), so rrtmgp and the dynamics scan compile as two small "
             "executables instead of one ~3h module. Requires --rad-update-steps>1. "
             "Default ON (needed to make the rrtmgp default compile in minutes; "
             "~1e-6 phase-shift vs the fused path). Pass --no-unfused-radiation "
             "for the byte-identical legacy fused path.",
    )
    # Atmosphere physics suite.  DEFAULT = full realistic CMIP6 atmosphere:
    # convection=sbm, turbulence=holtslag_boville, gravity-wave-drag=hines,
    # clouds=sundqvist, microphysics=kessler (+ rrtmgp radiation above).  This
    # suite is empirically stable coupled at coarse res (C18/L20, dt<=450s).
    # Pass --minimal-physics (or the individual --<scheme> none flags) for a
    # cheap idealized run.
    parser.add_argument("--convection", default="sbm",
                        choices=["sbm", "dca", "kuo", "mass_flux", "edmf", "none"],
                        help="Convection scheme (default: sbm)")
    parser.add_argument("--turbulence", default="holtslag_boville",
                        choices=["smagorinsky", "louis", "tke", "holtslag_boville",
                                 "mynn25", "clubb", "edmf", "none"],
                        help="Boundary-layer turbulence scheme "
                             "(default: holtslag_boville)")
    parser.add_argument("--gravity-wave-drag", default="hines",
                        choices=["rayleigh", "lindzen", "mcfarlane", "hines",
                                 "prognostic_spectral", "e3sm_cam", "ml_emulator",
                                 "none"],
                        help="Gravity-wave-drag scheme (default: hines)")
    parser.add_argument("--clouds", default="sundqvist",
                        choices=["none", "sundqvist", "xu_randall", "resolved"],
                        help="Cloud-fraction scheme (default: sundqvist)")
    parser.add_argument("--microphysics", default="kessler",
                        help="Microphysics scheme (default: kessler — closes the "
                             "water budget so convective condensate precipitates; "
                             "microphysics='none' with active convection gives "
                             "pr=0 and a cloud-water trap)")
    parser.add_argument(
        "--minimal-physics", action="store_true",
        help="Override the full-physics defaults to a cheap idealized "
             "atmosphere: gray radiation, SBM convection only "
             "(no turbulence / GWD / clouds / microphysics). For fast "
             "aquaplanet / dynamical-core sanity runs.",
    )
    parser.add_argument("--diag-days", type=int, default=5,
                        help="Diagnostic interval [days] (default: 5)")

    # Ocean.  The coupled driver runs a thermodynamic SLAB ocean (no 3D
    # dynamics — that lives in the standalone OceanModel and is not yet wired
    # into the coupler).  DEFAULT = two_layer: a mixed layer + deep layer with
    # bulk vertical mixing and deep-layer restoring (a cold deep reservoir that
    # damps SST drift), the most ocean physics the coupled slab supports today.
    parser.add_argument("--ocean", default="two_layer",
                        choices=["fixed", "slab", "two_layer", "dynamic"],
                        help="Coupled ocean mode (default: two_layer slab). "
                             "'dynamic' = the prognostic 3D LatLonCGridOceanModel "
                             "stepped by the coupler on a SHARED lat-lon grid "
                             "(requires --grid latlon); slab/two_layer/fixed = "
                             "thermodynamic slab")
    parser.add_argument("--ocean-h-mix", type=float, default=50.0,
                        help="Slab ocean mixed-layer depth [m]")
    parser.add_argument("--grid", default="cubed_sphere",
                        choices=["cubed_sphere", "latlon"],
                        help="Atmosphere grid (default cubed_sphere); 'latlon' "
                             "is required for --ocean dynamic (shared grid)")
    parser.add_argument("--ocean-nlev", type=int, default=20,
                        help="3D ocean vertical levels (--ocean dynamic)")
    parser.add_argument("--ocean-dt", type=float, default=300.0,
                        help="3D ocean SUBSTEP dt [s] (--ocean dynamic); the "
                             "coupler substeps the ocean at this dt within each "
                             "coupling_dt (never step the 3D ocean at 3600 s)")
    parser.add_argument("--ocean-H-max", type=float, default=5500.0,
                        help="Max ocean depth [m] (--ocean dynamic)")

    # Carbon
    parser.add_argument("--co2-init", type=float, default=415.0,
                        help="Initial CO2 concentration [ppmv]")

    # CMIP6 experiment / output
    parser.add_argument(
        "--experiment", default="",
        help="CMIP6 experiment id (e.g. historical, ssp585, piControl, "
             "1pctCO2). Selects the transient external-forcing trajectory "
             "(GHG/ozone/aerosol/solar). Empty = idealized/constant (default).",
    )
    parser.add_argument(
        "--start-year", type=int, default=1979,
        help="Calendar start year used to index CMIP6 forcing "
             "(e.g. 1850 for historical) (default: 1979)",
    )
    parser.add_argument(
        "--cmip-output", action="store_true",
        help="Write CMOR-style monthly NetCDF output (tas, pr, tos, siconc, ...)",
    )
    parser.add_argument(
        "--cmip-resolution-deg", type=float, default=5.0,
        help="Lat-lon grid spacing for CMIP output [deg] (default: 5.0)",
    )

    # Devices
    parser.add_argument(
        "--n-devices", type=int, default=None, metavar="N",
        help="Number of GPUs to use (default: auto-select largest valid count)",
    )

    # Output
    parser.add_argument("--output", "-o", default="results/coupled",
                        help="Output directory")

    args = parser.parse_args()

    # --minimal-physics: collapse the full-physics defaults to a cheap
    # idealized atmosphere (gray radiation + SBM convection only).  Applied
    # AFTER parsing so it cleanly overrides whatever the per-scheme defaults
    # are, without fighting argparse precedence.
    if args.minimal_physics:
        args.radiation = "gray"
        args.turbulence = "none"
        args.gravity_wave_drag = "none"
        args.clouds = "none"
        args.microphysics = "none"
        args.unfused_radiation = False
        args.rad_update_steps = 1
        args.ocean = "slab"          # cheap single-layer slab for idealized runs

    # Unfused radiation only engages when rad_update_steps > 1 (the host-loop
    # dispatch in _run_compiled requires it).  Make the no-op EXPLICIT rather
    # than silently falling back to the fused path.
    if args.unfused_radiation and args.rad_update_steps <= 1:
        logger.warning(
            "--unfused-radiation requires --rad-update-steps>1; got %d. "
            "Disabling unfused radiation (would silently no-op).",
            args.rad_update_steps,
        )
        args.unfused_radiation = False

    logger.info("=" * 60)
    logger.info("  legoESM Coupled ESM")
    logger.info("=" * 60)
    logger.info(f"  Preset:     {args.preset}")
    logger.info(f"  Resolution: C{args.resolution}/L{args.nlev}")
    logger.info(f"  Days:       {args.days}")
    logger.info(f"  Radiation:  {args.radiation}"
                + (" (unfused)" if args.unfused_radiation else ""))
    _full_suite = (
        args.radiation == "rrtmgp" and args.convection != "none"
        and args.turbulence != "none" and args.gravity_wave_drag != "none"
        and args.clouds != "none" and args.microphysics != "none"
    )
    _suite_tag = (
        "  [full CMIP6 suite]" if _full_suite
        else "  [MINIMAL]" if args.minimal_physics else "  [custom]"
    )
    logger.info(
        "  Physics:    conv=%s turb=%s gwd=%s clouds=%s micro=%s%s"
        % (args.convection, args.turbulence, args.gravity_wave_drag,
           args.clouds, args.microphysics, _suite_tag)
    )
    logger.info(f"  Ocean:      slab/{args.ocean}"
                + ("  (deep restoring)" if args.ocean == "two_layer" else ""))
    logger.info(f"  Experiment: {args.experiment or '(idealized/constant)'}"
                f"  start_year={args.start_year}")
    logger.info(f"  CMIP out:   {args.cmip_output}"
                + (f" @ {args.cmip_resolution_deg}deg" if args.cmip_output else ""))
    logger.info(f"  Devices:    {args.n_devices if args.n_devices is not None else 'auto'}")
    logger.info(f"  Backend:    {jax.default_backend()}")
    logger.info(f"  X64:        {jax.config.jax_enable_x64}")
    logger.info("=" * 60)

    # Build configs
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.ocean.simple_ocean import SimpleOceanConfig

    atm_config = ExperimentConfig(
        grid=GridConfig(
            grid_type=args.grid,
            resolution=args.resolution,
            nlev=args.nlev,
        ),
        dycore=DycoreConfig(
            dt=args.dt, model_type="hydrostatic",
            # On lat-lon, use the Arakawa-C-grid hydrostatic dycore so the atm
            # co-locates with the C-grid 3D ocean (--ocean dynamic); cdgrid is
            # cube-only.  Cube keeps the default cdgrid.
            discretization=("latlon_cgrid" if args.grid == "latlon"
                            else "cdgrid"),
        ),
        output=OutputConfig(
            diag_days=args.diag_days,
            cmip_output=args.cmip_output,
            cmip_resolution_deg=args.cmip_resolution_deg,
        ),
        radiation=args.radiation,
        rad_update_steps=args.rad_update_steps,
        unfused_radiation=args.unfused_radiation,
        convection=args.convection,
        turbulence=args.turbulence,
        gravity_wave_drag=args.gravity_wave_drag,
        cloud_scheme=args.clouds,
        microphysics=args.microphysics,
        days=args.days,
        experiment=args.experiment,
        start_year=args.start_year,
        n_devices=args.n_devices if args.n_devices is not None else "auto",
    )

    # Build coupled config from preset with overrides.  The ocean_config is
    # ALWAYS overridden from --ocean so the coupled default is the two_layer
    # slab (the presets all set a single-layer mode="slab"); two_layer enables
    # deep-layer restoring (a cold reservoir that damps SST drift) — the most
    # ocean physics the coupled slab supports.  Every preset holds a
    # SimpleOceanConfig, so replacing it is type-safe.
    overrides = {}
    if args.ocean == "dynamic":
        # Prognostic 3D LatLonCGridOceanModel on the SHARED lat-lon grid.
        from legoesm.ocean.state import LatLonCGridOceanConfig
        if args.grid != "latlon":
            raise SystemExit(
                "--ocean dynamic requires --grid latlon (the 3D ocean shares "
                "the atmosphere's lat-lon grid; cube-atm + tripole-ocean needs "
                "the deferred cross-grid remap).")
        overrides["ocean_mode"] = "dynamic"
        overrides["ocean_config"] = LatLonCGridOceanConfig()
        overrides["ocean_nlev"] = args.ocean_nlev
        overrides["ocean_dt_s"] = args.ocean_dt
        overrides["ocean_H_max_m"] = args.ocean_H_max
    elif args.ocean == "two_layer":
        overrides["ocean_config"] = SimpleOceanConfig(
            mode="two_layer", h_mix=args.ocean_h_mix, restore_deep=True,
        )
        overrides["ocean_mode"] = "two_layer"
    else:
        overrides["ocean_config"] = SimpleOceanConfig(
            mode=args.ocean, h_mix=args.ocean_h_mix,
        )
        # ocean_mode log label (fixed/slab -> "slab").
        overrides["ocean_mode"] = "slab"
    if args.co2_init != 415.0:
        overrides["co2_ppmv_init"] = args.co2_init

    coupled_cfg = PRESETS[args.preset](**overrides)

    # Create and run driver
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    driver = CoupledESMDriver(
        atm_config, coupled_cfg, output_dir=args.output,
    )

    t0 = time.time()
    driver.setup()
    t_setup = time.time() - t0
    logger.info(f"Setup completed in {t_setup:.1f}s")

    t0 = time.time()
    status = driver.run()
    t_run = time.time() - t0

    # Summary
    logger.info("=" * 60)
    logger.info(f"  Status: {status}")
    logger.info(f"  Wall time: {t_run:.1f}s ({t_run/60:.1f} min)")
    logger.info(f"  Per sim-day: {t_run / max(args.days, 1):.1f}s")

    # Final SST: slab stores T_sfc [K]; the dynamic 3D ocean stores top-level T
    # [degC] -> convert.  Use the driver's grid-agnostic accessor.
    sst = driver._ocean_surface_KuvC()[0]
    logger.info(f"  SST final: mean={float(sst.mean()):.1f}K, "
                f"range=[{float(sst.min()):.1f}, {float(sst.max()):.1f}]K")

    if driver.coupled_diagnostics:
        d0 = driver.coupled_diagnostics[0]
        df = driver.coupled_diagnostics[-1]
        drift = df["sst_mean"] - d0["sst_mean"]
        logger.info(f"  SST drift: {drift:.2f}K over {args.days} days "
                    f"({drift / max(args.days, 1) * 365:.1f} K/yr)")
        if "co2_ppmv_mean" in df:
            logger.info(f"  CO2: {df['co2_ppmv_mean']:.1f} ppmv")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
