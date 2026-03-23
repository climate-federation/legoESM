"""Diagnostics for 4D-Var data assimilation.

Innovation statistics, cost function monitoring, gradient norms.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp

from legoesm.da.control_vector import control_to_state, state_to_control
from legoesm.da.minimizer import MinimizationResult

logger = logging.getLogger(__name__)


def compute_innovation_statistics(
    state,
    observations: tuple,
    model,
    dt: float,
    n_steps: int,
    control_spec,
    template_state,
) -> dict:
    """Compute innovation (y - H(x)) statistics.

    Parameters
    ----------
    state : NamedTuple
        Current analysis or background state.
    observations : tuple of Observation
        Observation batches.
    model : object
        Model with .step(state, dt).
    dt : float
        Time step.
    n_steps : int
        Number of steps in window.
    control_spec : ControlVectorSpec
    template_state : NamedTuple

    Returns
    -------
    dict with keys:
        'innovation_mean': mean innovation
        'innovation_rms': RMS innovation
        'chi_squared': sum d^T R^{-1} d / n_obs (should be ~1 for optimal DA)
        'n_obs': total observation count
    """
    # Forward integration
    def scan_step(carry, _):
        s_new = model.step(carry, dt)
        return s_new, s_new

    _, trajectory = jax.lax.scan(scan_step, state, jnp.arange(n_steps))

    total_innovation = 0.0
    total_sq_innovation = 0.0
    total_chi2 = 0.0
    total_n_obs = 0

    for obs in observations:
        state_t = jax.tree.map(lambda arr: arr[obs.time_index], trajectory)
        H_x = obs.operator(state_t)
        d = obs.values - H_x
        n_obs = d.shape[0]

        total_innovation += float(jnp.sum(d))
        total_sq_innovation += float(jnp.sum(d ** 2))
        total_chi2 += float(jnp.sum(d ** 2 / obs.errors ** 2))
        total_n_obs += n_obs

    if total_n_obs == 0:
        return {
            "innovation_mean": 0.0,
            "innovation_rms": 0.0,
            "chi_squared": 0.0,
            "n_obs": 0,
        }

    return {
        "innovation_mean": total_innovation / total_n_obs,
        "innovation_rms": (total_sq_innovation / total_n_obs) ** 0.5,
        "chi_squared": total_chi2 / total_n_obs,
        "n_obs": total_n_obs,
    }


def log_minimization_progress(
    result: MinimizationResult,
    cycle: int = 0,
    outer: int = 0,
) -> None:
    """Log minimization progress."""
    logger.info(
        f"Cycle {cycle}, Outer {outer}: "
        f"J={float(result.fun):.6e}, "
        f"||grad||={float(result.grad_norm):.6e}, "
        f"iters={int(result.n_iter)}, "
        f"converged={bool(result.converged)}"
    )
