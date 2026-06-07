"""Incremental 4D-Var (Courtier, Thepaut & Hollingsworth 1994).

Outer loop linearizes around a trajectory; inner loop minimizes a
quadratic approximation. The adjoint is computed automatically by
jax.grad through the inner loop cost function.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.da.control_vector import (
    ControlVectorSpec,
    control_to_state,
    state_to_control,
)
from legoesm.da.cost_function import build_cost_fn
from legoesm.da.minimizer import minimize_cg, minimize_lbfgs
from legoesm.da.preconditioning import preconditioned_cost_fn

logger = logging.getLogger(__name__)


class IncrementalConfig(NamedTuple):
    """Configuration for incremental 4D-Var."""
    n_outer: int = 3
    n_inner: int = 50
    inner_gtol: float = 1e-5
    inner_method: str = "cg"       # "cg" or "lbfgs"
    use_preconditioning: bool = True
    checkpoint: bool = True


class IncrementalDiagnostics(NamedTuple):
    """Diagnostics from incremental 4D-Var."""
    cost_history: list
    grad_norm_history: list
    inner_iterations: list
    innovation_rms: list


def incremental_4dvar(
    model,
    background_state,
    observations: tuple,
    B,
    control_spec: ControlVectorSpec,
    dt: float,
    n_steps: int,
    config: IncrementalConfig = IncrementalConfig(),
    grid=None,
) -> tuple:
    """Run incremental 4D-Var.

    Outer loop (Python, not JIT):
    1. Linearize: build cost around current guess
    2. Inner loop (JIT): minimize cost using CG or L-BFGS
    3. Update: x^{k+1} = minimizer result

    Parameters
    ----------
    model : object
        Model with .step(state, dt).
    background_state : NamedTuple
        Background state.
    observations : tuple of Observation
        Observation batches.
    B : background error covariance
        Must have .inv_multiply() and .sqrt_multiply().
    control_spec : ControlVectorSpec
        Control vector specification.
    dt : float
        Model time step [s].
    n_steps : int
        Assimilation window length in steps.
    config : IncrementalConfig
        Configuration.
    grid : GridProtocol, optional
        Grid.

    Returns
    -------
    (analysis_state, IncrementalDiagnostics)
    """
    x_b = state_to_control(background_state, control_spec)
    x_k = x_b.copy()

    cost_history = []
    grad_norm_history = []
    inner_iterations = []
    innovation_rms_list = []

    for outer in range(config.n_outer):
        # Build cost function around current iterate
        template = control_to_state(x_k, control_spec, background_state)

        cost_fn = build_cost_fn(
            model, x_b, observations, B, control_spec,
            template, dt, n_steps, config.checkpoint,
        )
        cost_and_grad = jax.value_and_grad(cost_fn)

        # Compute current cost and gradient
        J_k, g_k = cost_and_grad(x_k)
        g_norm = float(jnp.linalg.norm(g_k))
        cost_history.append(float(J_k))
        grad_norm_history.append(g_norm)

        logger.info(
            f"Outer {outer}: J={float(J_k):.6e}, ||grad||={g_norm:.6e}"
        )

        if g_norm < config.inner_gtol:
            inner_iterations.append(0)
            innovation_rms_list.append(0.0)
            continue

        # Inner loop minimization
        if config.use_preconditioning:
            J_tilde = preconditioned_cost_fn(cost_fn, B, x_b)
            J_tilde_and_grad = jax.value_and_grad(J_tilde)
            # Initial v from current x: x_k = x_b + B^{1/2} v
            v0 = jnp.zeros_like(x_k)

            if config.inner_method == "cg":
                result = minimize_cg(
                    jax.jit(J_tilde_and_grad), v0,
                    max_iter=config.n_inner, gtol=config.inner_gtol,
                )
            else:
                result = minimize_lbfgs(
                    jax.jit(J_tilde_and_grad), v0,
                    max_iter=config.n_inner, gtol=config.inner_gtol,
                )

            # Recover x from v
            x_k = x_b + B.sqrt_multiply(result.x)
        else:
            if config.inner_method == "cg":
                result = minimize_cg(
                    jax.jit(cost_and_grad), x_k,
                    max_iter=config.n_inner, gtol=config.inner_gtol,
                )
            else:
                result = minimize_lbfgs(
                    jax.jit(cost_and_grad), x_k,
                    max_iter=config.n_inner, gtol=config.inner_gtol,
                )
            x_k = result.x

        inner_iterations.append(int(result.n_iter))

        # Compute innovation RMS
        J_final = float(result.fun)
        innovation_rms_list.append(J_final)

    analysis_state = control_to_state(x_k, control_spec, background_state)

    diagnostics = IncrementalDiagnostics(
        cost_history=cost_history,
        grad_norm_history=grad_norm_history,
        inner_iterations=inner_iterations,
        innovation_rms=innovation_rms_list,
    )

    return analysis_state, diagnostics
