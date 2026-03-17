#!/usr/bin/env python
"""AMIP simulation — thin CLI wrapper around ModelDriver.

Usage:
    JAX_ENABLE_X64=1 python scripts/run_amip.py \
        --dataset analytical --days 30 --resolution 16 --dt 600

    JAX_ENABLE_X64=1 python scripts/run_amip.py \
        --dataset cobe --forcing-path /path/to/MODEL.SST.COBE-SST2.nc \
        --days 365 --resolution 24 --dt 450 \
        --checkpoint-days 30 --output results/amip_1yr

    # Restart from checkpoint:
    JAX_ENABLE_X64=1 python scripts/run_amip.py \
        --restart-from results/amip_1yr/checkpoint_day_030.npz \
        --forcing-path /path/to/COBE.nc --days 365
"""

import argparse
import sys

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="AMIP simulation with prescribed SST/SIC",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
# Forcing
parser.add_argument("--dataset", type=str, default="analytical",
                    choices=["cobe", "hadisst", "custom", "analytical"])
parser.add_argument("--forcing-path", type=str, default=None)
parser.add_argument("--sst-var", type=str, default=None)
parser.add_argument("--sic-var", type=str, default=None)
parser.add_argument("--sst-offset", type=float, default=None)
parser.add_argument("--sic-scale", type=float, default=None)
# Grid
parser.add_argument("--resolution", type=int, default=16)
parser.add_argument("--nlev", type=int, default=40)
parser.add_argument("--vertical-coord", type=str, default="hybrid",
                    choices=["sigma", "hybrid"])
parser.add_argument("--p-top", type=float, default=None)
parser.add_argument("--stretching", type=float, default=None)
parser.add_argument("--discretization", type=str, default="centered",
                    choices=["centered", "finite_volume", "cgrid"])
# Integration
parser.add_argument("--start-day", type=float, default=0.0)
parser.add_argument("--days", type=int, default=200)
parser.add_argument("--dt", type=float, default=600.0)
parser.add_argument("--diag-days", type=int, default=5)
# Output
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--checkpoint-days", type=int, default=0)
parser.add_argument("--restart-from", type=str, default=None)
# Radiation
parser.add_argument("--radiation", type=str, default="gray",
                    choices=["gray", "rrtmg", "rrtmgp"])
parser.add_argument("--rad-update-steps", type=int, default=1)
parser.add_argument("--diurnal-cycle", action="store_true", default=False)
parser.add_argument("--co2-ppmv", type=float, default=415.0)
parser.add_argument("--ch4-ppbv", type=float, default=1900.0)
parser.add_argument("--n2o-ppbv", type=float, default=332.0)
parser.add_argument("--ozone-source", type=str, default="standard",
                    choices=["standard", "analytical", "none"])
parser.add_argument("--ozone-forcing", type=str, default="inline",
                    choices=["inline", "external", "off"])
parser.add_argument("--ozone-file", type=str, default="")
# Solar
parser.add_argument("--solar-source", type=str, default="constant",
                    choices=["constant", "file", "spectral_file"])
parser.add_argument("--solar-file", type=str, default="")
parser.add_argument("--solar-spectral-var", type=str,
                    default="solar_fraction_by_gpt")
# Aerosol
parser.add_argument("--aerosol-forcing", type=str, default="off",
                    choices=["off", "external"])
parser.add_argument("--aerosol-file", type=str, default="")
parser.add_argument("--aerosol-reference-aod", type=float, default=0.03)
parser.add_argument("--volcanic-aerosol-file", type=str, default="")
parser.add_argument("--volcanic-aerosol-scale", type=float, default=1.0)
# Clouds & microphysics
parser.add_argument("--clouds", type=str, default="none",
                    choices=["none", "sundqvist", "xu_randall"])
parser.add_argument("--microphysics", type=str, default="none",
                    choices=["none", "kessler", "sundqvist", "seifert_beheng",
                             "morrison", "thompson"])
# Topography
parser.add_argument("--topography", type=str, default="flat")
parser.add_argument("--topo-smoothing", type=int, default=4)
parser.add_argument("--topo-edge-blend", type=float, default=0.3)
# Surface
parser.add_argument("--dynamic-albedo", action="store_true", default=False)
parser.add_argument("--monthly-means", action="store_true", default=False)
# Moisture conservation
parser.add_argument("--fix-moisture", action="store_true", default=False)
# CMIP
parser.add_argument("--experiment", type=str, default="")
parser.add_argument("--start-year", type=int, default=1979)
parser.add_argument("--cmip-output", action="store_true", default=False)
parser.add_argument("--clear-sky-diag", action="store_true", default=False)
# Distributed (placeholder — not yet wired through ModelDriver)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--ensemble-size", type=int, default=1)

args = parser.parse_args()

# Backward-compatible alias
if args.radiation == "rrtmgp":
    args.radiation = "rrtmg"

# Enforce forcing-path requirement
if (args.forcing_path is None and args.restart_from is None
        and args.dataset != "analytical"):
    parser.error("--forcing-path required (unless --dataset analytical or --restart-from)")
if args.solar_source in ("file", "spectral_file") and not args.solar_file:
    parser.error("--solar-file required when --solar-source is file/spectral_file")

# ---------------------------------------------------------------------------
# Build ExperimentConfig and run via ModelDriver
# ---------------------------------------------------------------------------
from legoesm.driver import (
    GridConfig, DycoreConfig, OutputConfig, ExperimentConfig, ModelDriver,
)

_p_top = args.p_top if args.p_top is not None else (
    200.0 if args.vertical_coord == "hybrid" else 1000.0
)
_stretching = args.stretching if args.stretching is not None else (
    2.0 if args.vertical_coord == "hybrid" else 0.0
)

config = ExperimentConfig(
    grid=GridConfig(
        grid_type="cubed_sphere",
        resolution=args.resolution,
        nlev=args.nlev,
        vertical_coord=args.vertical_coord,
        p_top_Pa=_p_top,
        stretching=_stretching,
    ),
    dycore=DycoreConfig(
        discretization=args.discretization,
        dt=args.dt,
    ),
    output=OutputConfig(
        output_dir=args.output or "",
        diag_days=args.diag_days,
        checkpoint_days=args.checkpoint_days,
        monthly_means=args.monthly_means,
        cmip_output=args.cmip_output,
        clear_sky_diag=args.clear_sky_diag,
    ),
    days=args.days,
    start_day=args.start_day,
    dataset=args.dataset,
    forcing_path=args.forcing_path or "",
    sst_var=args.sst_var or "",
    sic_var=args.sic_var or "",
    sst_offset=args.sst_offset if args.sst_offset is not None else 0.0,
    sic_scale=args.sic_scale if args.sic_scale is not None else 1.0,
    radiation=args.radiation,
    rad_update_steps=args.rad_update_steps,
    diurnal_cycle=args.diurnal_cycle,
    co2_ppmv=args.co2_ppmv,
    ch4_ppbv=args.ch4_ppbv,
    n2o_ppbv=args.n2o_ppbv,
    ozone_source=args.ozone_source,
    ozone_forcing=args.ozone_forcing,
    ozone_file=args.ozone_file,
    solar_source=args.solar_source,
    solar_file=args.solar_file,
    solar_spectral_var=args.solar_spectral_var,
    aerosol_forcing=args.aerosol_forcing,
    aerosol_file=args.aerosol_file,
    aerosol_reference_aod=args.aerosol_reference_aod,
    volcanic_aerosol_file=args.volcanic_aerosol_file,
    volcanic_aerosol_scale=args.volcanic_aerosol_scale,
    cloud_scheme=args.clouds,
    microphysics=args.microphysics,
    topography=args.topography,
    topo_smoothing=args.topo_smoothing,
    topo_edge_blend=args.topo_edge_blend,
    dynamic_albedo=args.dynamic_albedo,
    fix_moisture=args.fix_moisture,
    experiment=args.experiment,
    start_year=args.start_year,
    distributed=args.distributed,
    ensemble_size=args.ensemble_size,
)

print("=" * 60)
print(f"  AMIP: {config.radiation} + SBM, C{args.resolution}/L{args.nlev}")
print("=" * 60)

driver = ModelDriver(config, output_dir=args.output)
driver.setup()

# Handle restart
start_step = 0
start_day = None
if args.restart_from:
    start_step, start_day = driver.load_checkpoint(args.restart_from)
    print(f"  Restarting from step {start_step}, day {start_day:.1f}")

status = driver.run(start_step=start_step, start_day=start_day)

print(f"\n  All outputs saved to {driver.output_dir}/")
print("  Done.")
