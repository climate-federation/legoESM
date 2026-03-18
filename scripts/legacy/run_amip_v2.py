#!/usr/bin/env python
"""AMIP simulation using the composable ModelDriver.

Thin CLI wrapper around legoesm.driver.ModelDriver — ~100 lines vs 2100+
in run_amip.py.  Produces identical results for matching configurations.

Usage:
    JAX_ENABLE_X64=1 python scripts/run_amip_v2.py \
        --dataset analytical --days 30 --resolution 16 --dt 600

    JAX_ENABLE_X64=1 python scripts/run_amip_v2.py \
        --dataset cobe --forcing-path /path/to/cobe.nc \
        --days 365 --resolution 24 --dt 450 --checkpoint-days 30
"""

import argparse
import sys

sys.stdout.reconfigure(line_buffering=True)

from legoesm.driver import (
    GridConfig, DycoreConfig, OutputConfig, ExperimentConfig, ModelDriver,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="AMIP simulation (composable driver)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

# Grid / integration
parser.add_argument("--resolution", type=int, default=16)
parser.add_argument("--nlev", type=int, default=40)
parser.add_argument("--vertical-coord", type=str, default="hybrid",
                    choices=["sigma", "hybrid"])
parser.add_argument("--p-top", type=float, default=None)
parser.add_argument("--stretching", type=float, default=None)
parser.add_argument("--dt", type=float, default=600.0)
parser.add_argument("--days", type=int, default=200)
parser.add_argument("--start-day", type=float, default=0.0)
parser.add_argument("--discretization", type=str, default="centered",
                    choices=["centered", "finite_volume", "cgrid"])

# Forcing
parser.add_argument("--dataset", type=str, default="analytical",
                    choices=["cobe", "hadisst", "custom", "analytical"])
parser.add_argument("--forcing-path", type=str, default="")
parser.add_argument("--sst-var", type=str, default="")
parser.add_argument("--sic-var", type=str, default="")
parser.add_argument("--sst-offset", type=float, default=0.0)
parser.add_argument("--sic-scale", type=float, default=1.0)

# Physics
parser.add_argument("--radiation", type=str, default="gray",
                    choices=["gray", "rrtmg", "rrtmgp"])
parser.add_argument("--rad-update-steps", type=int, default=1)
parser.add_argument("--diurnal-cycle", action="store_true", default=False)
parser.add_argument("--co2-ppmv", type=float, default=415.0)
parser.add_argument("--ch4-ppbv", type=float, default=1900.0)
parser.add_argument("--n2o-ppbv", type=float, default=332.0)
parser.add_argument("--ozone-source", type=str, default="standard",
                    choices=["standard", "analytical", "none"])
parser.add_argument("--clouds", type=str, default="none",
                    choices=["none", "sundqvist", "xu_randall"])
parser.add_argument("--microphysics", type=str, default="none",
                    choices=["none", "kessler", "sundqvist",
                             "seifert_beheng", "morrison", "thompson"])

# Topography / surface
parser.add_argument("--topography", type=str, default="flat")
parser.add_argument("--topo-smoothing", type=int, default=4)
parser.add_argument("--topo-edge-blend", type=float, default=0.3)
parser.add_argument("--dynamic-albedo", action="store_true", default=False)

# Output / diagnostics
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--diag-days", type=int, default=5)
parser.add_argument("--checkpoint-days", type=int, default=0)
parser.add_argument("--monthly-means", action="store_true", default=False)
parser.add_argument("--cmip-output", action="store_true", default=False)
parser.add_argument("--clear-sky-diag", action="store_true", default=False)

# Restart
parser.add_argument("--restart-from", type=str, default=None)

args = parser.parse_args()

# Backward-compatible alias
if args.radiation == "rrtmgp":
    args.radiation = "rrtmg"

# ---------------------------------------------------------------------------
# Build ExperimentConfig
# ---------------------------------------------------------------------------
p_top = args.p_top if args.p_top is not None else (200.0 if args.nlev <= 40 else 10.0)
stretching = args.stretching if args.stretching is not None else 2.0

grid_cfg = GridConfig(
    resolution=args.resolution, nlev=args.nlev,
    vertical_coord=args.vertical_coord,
    p_top_Pa=p_top, stretching=stretching,
)
dycore_cfg = DycoreConfig(dt=args.dt, discretization=args.discretization)
output_cfg = OutputConfig(
    output_dir=args.output or "",
    diag_days=args.diag_days,
    checkpoint_days=args.checkpoint_days,
    monthly_means=args.monthly_means,
    cmip_output=args.cmip_output,
    clear_sky_diag=args.clear_sky_diag,
)

config = ExperimentConfig(
    grid=grid_cfg, dycore=dycore_cfg, output=output_cfg,
    days=args.days, start_day=args.start_day,
    dataset=args.dataset, forcing_path=args.forcing_path,
    sst_var=args.sst_var, sic_var=args.sic_var,
    sst_offset=args.sst_offset, sic_scale=args.sic_scale,
    radiation=args.radiation, rad_update_steps=args.rad_update_steps,
    diurnal_cycle=args.diurnal_cycle,
    co2_ppmv=args.co2_ppmv, ch4_ppbv=args.ch4_ppbv, n2o_ppbv=args.n2o_ppbv,
    ozone_source=args.ozone_source,
    cloud_scheme=args.clouds, microphysics=args.microphysics,
    topography=args.topography, topo_smoothing=args.topo_smoothing,
    topo_edge_blend=args.topo_edge_blend,
    dynamic_albedo=args.dynamic_albedo,
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print(f"=== AMIP v2 (ModelDriver) ===")
driver = ModelDriver(config, output_dir=args.output)
driver.setup()

if args.restart_from:
    step, day = driver.load_checkpoint(args.restart_from)
    print(f"  Restarted from step={step}, day={day:.1f}")
    status = driver.run(start_step=step, start_day=day)
else:
    status = driver.run()

print(f"\n  Status: {status}")
print(f"  Output: {driver.output_dir}")
