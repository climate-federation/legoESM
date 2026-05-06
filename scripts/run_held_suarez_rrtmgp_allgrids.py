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

    # iter-121 (codex iter-119-followup MEDIUM-3): capture
    # initial atmospheric mass for the post-run mass-drift
    # gate.  Pre-iter-121, this allgrids HS+RRTMGP runner only
    # checked ``driver.run()`` status (stability/blowup), not
    # conservation.  This made it pass even on dycore-side
    # mass-conservation regressions.
    mass_init = _compute_driver_mass(driver)

    t0 = time.time()
    status = driver.run(compiled=False)
    t_run = time.time() - t0

    # iter-121: compute final mass and apply the same 1e-2
    # tolerance the matrix runner uses for HS.  If we can't
    # compute mass for this state type, skip the gate (graceful
    # degradation — the status-based gate is still in effect).
    mass_final = _compute_driver_mass(driver)
    mass_drift = None
    if mass_init is not None and mass_final is not None:
        from legoesm.diagnostics.conservation_drift import compute_relative_drift
        mass_drift = compute_relative_drift([mass_init, mass_final])
        import numpy as _np
        # iter-124 (codex iter-123-followup LOW-2): only
        # override when the original status is ``COMPLETED``.
        # Pre-iter-124 the override also rewrote already-failed
        # statuses (e.g., ``BLOWUP at day 5``) into
        # ``FAIL: mass drift ... (was BLOWUP at day 5)``,
        # weakening the ``ModelDriver.run()`` status contract
        # and breaking consumers that classify by prefix.
        if (status == "COMPLETED" and
                (not _np.isfinite(mass_drift) or mass_drift > 1e-2)):
            status = (
                f"FAIL: mass drift {mass_drift:.2e} exceeds "
                f"tolerance 1e-2"
            )

    logger.info(
        f"{name}: {status} (setup={t_setup:.1f}s, "
        f"run={t_run:.1f}s, mass_drift="
        f"{'N/A' if mass_drift is None else f'{mass_drift:.2e}'})"
    )

    return {
        "name": name,
        "status": status,
        "setup_time": t_setup,
        "run_time": t_run,
        "mass_drift": mass_drift,
    }


def _compute_driver_mass(driver) -> float | None:
    """Compute total atmospheric mass ∫p_s dA from a
    ``ModelDriver`` (driver holds both state and grid).

    iter-121: walks common ``driver.state.p_s.data`` / area
    attribute paths.  Returns None for state layouts where
    we can't find a (p_s, area) pair (e.g., spectral states
    that store ``lnps_hat`` instead).  None disables the
    iter-121 gate gracefully — the status-based BLOWUP gate
    from ``ModelDriver.run()`` is still in effect.

    Returns
    -------
    Total mass-related integral ``∫p_s dA`` in Pa·m², or
    ``None`` when the layout is not recognized.
    """
    import jax.numpy as _jnp
    state = getattr(driver, "state", None)
    if state is None:
        return None
    p_s = getattr(getattr(state, "p_s", None), "data", None)
    if p_s is None:
        return None
    # Driver holds the grid object; try common area-attribute
    # paths in priority order.
    grid = getattr(driver, "grid", None)
    if grid is None:
        return None
    for attr in ("area", "areaCell", "grid_area"):
        area = getattr(grid, attr, None)
        if area is not None:
            try:
                return float(_jnp.sum(p_s * area))
            except Exception:
                return None
    return None


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
