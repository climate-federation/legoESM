"""Canonical Held-Suarez on icosahedral (SCVT) ~0.5 degree for 100 days.

subdivision_level=7 → 163,842 cells, mean cell spacing ≈ 62 km (~0.56°).
SCVT mesh is generated from scratch with 50 Lloyd iterations inside
ModelDriver.setup(); this step is the main wall-clock cost before
integration begins.

Runs on whatever JAX backend is available (single GPU fallback on this
host — no MPI). dt=300 s (halved from the 600 s used at lower resolution)
for gravity-wave CFL headroom at 62 km cell spacing.

No real radiation: the Held-Suarez 1994 benchmark is defined by
Newtonian relaxation to T_eq + Rayleigh friction, so radiation='none'
is canonical. Attempting RRTMGP at 163k columns × 60 g-points × 20
levels × fp64 exceeded 24 GB on the single GPU here.

Usage:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_held_suarez_icos_0p5deg.py
"""
from __future__ import annotations

import logging
import time

import jax

from legoesm.driver.model_driver import ModelDriver
from legoesm.driver.config import (
    ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("held_suarez_icos_0p5deg")

N_DAYS = 100
DIAG_DAYS = 10
SUBDIVISION_LEVEL = 7  # ≈0.56° mean spacing, 163,842 cells
DT_SECONDS = 300.0


def main():
    logger.info("JAX devices: %s (backend=%s)",
                jax.devices(), jax.default_backend())

    config = ExperimentConfig(
        grid=GridConfig(
            grid_type="voronoi",
            resolution=SUBDIVISION_LEVEL,
            nlev=20,
            vertical_coord="sigma",
        ),
        dycore=DycoreConfig(
            model_type="hydrostatic",
            discretization="mpas",
            dt=DT_SECONDS,
            hyperdiff_scale=1.0,
            div_damp_scale=1.0,
            conservation_fixer=True,
            fix_mass=True,
        ),
        output=OutputConfig(
            output_dir=f"results/held_suarez_icos_L{SUBDIVISION_LEVEL}_0p5deg",
            diag_days=DIAG_DAYS,
        ),
        days=N_DAYS,
        dataset="analytical",
        radiation="none",
        rad_update_steps=1,
        diurnal_cycle=False,
        convection="none",
        turbulence="none",
        gravity_wave_drag="none",
        microphysics="none",
        cloud_scheme="none",
        topography="flat",
        T_init=300.0,
        rh_init=0.0,
        precision="fp64",
        held_suarez_forcing=True,
    )

    logger.info("Starting canonical Held-Suarez on icosahedral SCVT "
                "(%d days, subdivision_level=%d, dt=%.0fs, radiation=none)",
                N_DAYS, SUBDIVISION_LEVEL, DT_SECONDS)

    driver = ModelDriver(config)

    t0 = time.time()
    driver.setup()
    t_setup = time.time() - t0
    logger.info("setup done in %.1f s (mesh-gen dominates)", t_setup)

    t0 = time.time()
    status = driver.run(compiled=False)
    t_run = time.time() - t0

    logger.info("status=%s  setup=%.1fs  run=%.1fs", status, t_setup, t_run)

    # iter-109 (codex iter-104 MEDIUM-8): propagate status to
    # exit code so wrappers / automation can detect BLOWUP via
    # ``$?``.
    from legoesm.driver.run_status import status_to_exit_code
    return status_to_exit_code(status)


if __name__ == "__main__":
    import sys
    sys.exit(main())
