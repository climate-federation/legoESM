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
    parser.add_argument("--radiation", default="gray",
                        choices=["gray", "rrtmg", "rrtmgp"],
                        help="Radiation scheme (default: gray)")
    parser.add_argument("--diag-days", type=int, default=5,
                        help="Diagnostic interval [days] (default: 5)")

    # Ocean
    parser.add_argument("--ocean-h-mix", type=float, default=50.0,
                        help="Slab ocean mixed-layer depth [m]")

    # Carbon
    parser.add_argument("--co2-init", type=float, default=415.0,
                        help="Initial CO2 concentration [ppmv]")

    # Devices
    parser.add_argument(
        "--n-devices", type=int, default=None, metavar="N",
        help="Number of GPUs to use (default: auto-select largest valid count)",
    )

    # Output
    parser.add_argument("--output", "-o", default="results/coupled",
                        help="Output directory")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  legoESM Coupled ESM")
    logger.info("=" * 60)
    logger.info(f"  Preset:     {args.preset}")
    logger.info(f"  Resolution: C{args.resolution}/L{args.nlev}")
    logger.info(f"  Days:       {args.days}")
    logger.info(f"  Radiation:  {args.radiation}")
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
            grid_type="cubed_sphere",
            resolution=args.resolution,
            nlev=args.nlev,
        ),
        dycore=DycoreConfig(dt=args.dt, model_type="hydrostatic"),
        output=OutputConfig(diag_days=args.diag_days),
        radiation=args.radiation,
        days=args.days,
        n_devices=args.n_devices if args.n_devices is not None else "auto",
    )

    # Build coupled config from preset with overrides
    overrides = {}
    if args.ocean_h_mix != 50.0:
        overrides["ocean_config"] = SimpleOceanConfig(
            mode="slab", h_mix=args.ocean_h_mix,
        )
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

    sst = driver.ocean_state.T_sfc.data
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
