"""Assimilation cycling: sequential analysis-forecast loops.

Each cycle runs incremental 4D-Var on the current window, then
integrates the analysis forward to the start of the next window
as the new background.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.da.control_vector import ControlVectorSpec, state_to_control, control_to_state
from legoesm.da.incremental import IncrementalConfig, incremental_4dvar

logger = logging.getLogger(__name__)


class CyclingConfig(NamedTuple):
    """Configuration for DA cycling."""
    window_length: int         # Assimilation window [time steps]
    cycle_length: int          # Forecast between windows [time steps]
    dt: float                  # Model time step [s]
    n_cycles: int = 1
    incremental: IncrementalConfig = IncrementalConfig()


def run_cycling(
    model,
    initial_state,
    observations_per_cycle: tuple,
    B,
    control_spec: ControlVectorSpec,
    config: CyclingConfig,
    grid=None,
) -> tuple:
    """Run sequential DA cycling.

    For each cycle:
    1. Analysis: run incremental 4D-Var on current window
    2. Forecast: integrate analysis forward to start of next window
    3. Use forecast as background for next cycle

    Parameters
    ----------
    model : object
        Model with .step(state, dt).
    initial_state : NamedTuple
        Initial background state for first cycle.
    observations_per_cycle : tuple of tuple of Observation
        Observations for each cycle.
    B : background error covariance
    control_spec : ControlVectorSpec
    config : CyclingConfig
    grid : GridProtocol, optional

    Returns
    -------
    (final_analysis_state, list of per-cycle diagnostics)
    """
    background = initial_state
    cycle_diagnostics = []

    for cycle in range(config.n_cycles):
        logger.info(f"=== Cycle {cycle + 1}/{config.n_cycles} ===")

        obs = observations_per_cycle[cycle]

        # Run 4D-Var analysis
        analysis, diag = incremental_4dvar(
            model=model,
            background_state=background,
            observations=obs,
            B=B,
            control_spec=control_spec,
            dt=config.dt,
            n_steps=config.window_length,
            config=config.incremental,
            grid=grid,
        )
        cycle_diagnostics.append(diag)

        # Forecast to next cycle (if not last cycle)
        if cycle < config.n_cycles - 1:
            forecast_steps = config.cycle_length

            def scan_step(carry, _):
                s_new = model.step(carry, config.dt)
                return s_new, None

            background, _ = jax.lax.scan(
                scan_step, analysis, jnp.arange(forecast_steps)
            )
        else:
            background = analysis

    return background, cycle_diagnostics
