"""Run Held-Suarez + RRTMGP on all grid types for 100 days.

Compares cubed-sphere, lat-lon FV, spectral (Gaussian), and icosahedral
(MPAS) grids for physical consistency and stability.

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python scripts/run_held_suarez_rrtmgp_allgrids.py
"""
from __future__ import annotations

import logging
import sys
import time

from legoesm.driver.model_driver import ModelDriver
from legoesm.driver.config import (
    ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("held_suarez_allgrids")

N_DAYS = 100
DIAG_DAYS = 10

GRID_CONFIGS = {
    "cubed_sphere": dict(
        grid=GridConfig(
            grid_type="cubed_sphere", resolution=16, nlev=20,
            vertical_coord="hybrid", p_top_Pa=200.0, stretching=2.0,
        ),
        dycore=DycoreConfig(
            model_type="hydrostatic", discretization="cdgrid",
            dt=600.0, hyperdiff_scale=1.0, div_damp_scale=1.0,
            conservation_fixer=True, fix_mass=True,
        ),
    ),
    "latlon_fv": dict(
        grid=GridConfig(
            grid_type="latlon", resolution=48, nlev=20,
            vertical_coord="hybrid", p_top_Pa=200.0, stretching=2.0,
        ),
        dycore=DycoreConfig(
            model_type="hydrostatic", discretization="latlon_fv",
            dt=600.0, hyperdiff_scale=1.0, div_damp_scale=1.0,
            conservation_fixer=True, fix_mass=True,
        ),
    ),
    "spectral": dict(
        grid=GridConfig(
            grid_type="gaussian", resolution=21, nlev=20,
            vertical_coord="hybrid", p_top_Pa=200.0, stretching=2.0,
        ),
        dycore=DycoreConfig(
            model_type="hydrostatic", discretization="spectral",
            dt=900.0, hyperdiff_scale=1.0, div_damp_scale=1.0,
        ),
    ),
    "icosahedral": dict(
        grid=GridConfig(
            grid_type="voronoi", resolution=3, nlev=20,
            vertical_coord="sigma",
        ),
        dycore=DycoreConfig(
            model_type="hydrostatic", discretization="mpas",
            dt=600.0, hyperdiff_scale=1.0, div_damp_scale=1.0,
            conservation_fixer=True, fix_mass=True,
        ),
    ),
}


def run_grid(name: str, grid_kwargs: dict) -> dict:
    """Run a single grid type and return diagnostics summary."""
    logger.info(f"{'='*60}")
    logger.info(f"Starting {name} ({N_DAYS} days)")
    logger.info(f"{'='*60}")

    config = ExperimentConfig(
        **grid_kwargs,
        output=OutputConfig(
            output_dir=f"results/held_suarez_rrtmgp_{name}",
            diag_days=DIAG_DAYS,
        ),
        days=N_DAYS,
        dataset="analytical",
        radiation="rrtmgp",
        rad_update_steps=3,
        diurnal_cycle=False,
        convection="none",
        turbulence="none",
        gravity_wave_drag="none",
        microphysics="none",
        cloud_scheme="none",
        topography="flat",
        T_init=300.0,
        RH_init=0.0,
        precision="fp64",
        held_suarez_forcing=True,
    )

    driver = ModelDriver(config)
    t0 = time.time()
    driver.setup()
    t_setup = time.time() - t0

    t0 = time.time()
    status = driver.run(compiled=False)
    t_run = time.time() - t0

    logger.info(f"{name}: {status} (setup={t_setup:.1f}s, run={t_run:.1f}s)")

    return {
        "name": name,
        "status": status,
        "setup_time": t_setup,
        "run_time": t_run,
    }


def main():
    results = {}
    for name, kwargs in GRID_CONFIGS.items():
        try:
            results[name] = run_grid(name, kwargs)
        except Exception as e:
            logger.error(f"{name}: FAILED with {type(e).__name__}: {e}")
            results[name] = {"name": name, "status": f"FAILED: {e}"}

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    for name, r in results.items():
        logger.info(f"  {name:20s}: {r['status']}")

    # iter-109 (codex iter-104 MEDIUM-8): propagate per-grid
    # status to exit code so wrappers / automation can detect
    # BLOWUP via ``$?``.  Returns 0 if all grids are
    # ``COMPLETED``, 1 if any grid hit BLOWUP / FAILED /
    # exception (caught above as ``FAILED: ...``).
    from legoesm.driver.run_status import status_to_exit_code
    overall = max(
        status_to_exit_code(r["status"]) for r in results.values()
    )
    return overall


if __name__ == "__main__":
    import sys
    sys.exit(main())
