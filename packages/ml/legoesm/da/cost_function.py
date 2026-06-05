"""4D-Var cost function J(x) = J_b + J_o.

The cost function is a pure function of the control vector, differentiable
by jax.grad. Forward integration uses jax.lax.scan with optional
gradient checkpointing for memory efficiency.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from legoesm.da.control_vector import control_to_state


def build_cost_fn(
    model,
    background: jax.Array,
    observations: tuple,
    B,
    control_spec,
    template_state,
    dt: float,
    n_steps: int,
    checkpoint: bool = True,
) -> Callable[[jax.Array], jax.Array]:
    """Build a JIT-compilable 4D-Var cost function.

    Returns J: control_vector -> scalar

    J(x) = 1/2 (x - x_b)^T B^{-1} (x - x_b)
          + 1/2 sum_i (y_i - H_i(M_i(x)))^T R_i^{-1} (y_i - H_i(M_i(x)))

    Parameters
    ----------
    model : object
        Model with .step(state, dt) method.
    background : jax.Array
        Background state in control space.
    observations : tuple of Observation
        Observation batches with time_index, values, errors, operator.
    B : background error covariance
        Must have .inv_multiply(x) method.
    control_spec : ControlVectorSpec
        Control vector specification.
    template_state : NamedTuple
        Template state for static fields.
    dt : float
        Model time step [s].
    n_steps : int
        Number of time steps in assimilation window.
    checkpoint : bool
        If True (default), apply ``jax.checkpoint`` to the scan body
        so intermediates are recomputed during the backward pass,
        trading compute for memory.  Set False to store all
        intermediates (faster but O(n_steps) memory).
    """

    def cost_fn(x: jax.Array) -> jax.Array:
        # Background term
        dx = x - background
        J_b = 0.5 * jnp.sum(dx * B.inv_multiply(dx))

        # Forward model integration
        state_0 = control_to_state(x, control_spec, template_state)

        def scan_step(carry, _):
            s_new = model.step(carry, dt)
            return s_new, s_new

        # Apply gradient checkpointing to reduce memory
        if checkpoint:
            scan_step_ckpt = jax.checkpoint(
                scan_step,
                prevent_cse=False,
            )
        else:
            scan_step_ckpt = scan_step

        _, trajectory = jax.lax.scan(
            scan_step_ckpt, state_0, jnp.arange(n_steps)
        )

        # Observation term — match dtype of background term
        J_o = jnp.zeros((), dtype=J_b.dtype)
        for obs in observations:
            state_t = jax.tree.map(lambda arr: arr[obs.time_index], trajectory)
            H_x = obs.operator(state_t)
            d = obs.values - H_x
            R_inv_d = d / (obs.errors ** 2)
            J_o = J_o + 0.5 * jnp.sum(d * R_inv_d)

        return J_b + J_o

    return cost_fn


def build_cost_and_grad_fn(
    model,
    background: jax.Array,
    observations: tuple,
    B,
    control_spec,
    template_state,
    dt: float,
    n_steps: int,
    checkpoint: bool = True,
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    """Build JIT-compiled (J, nabla J) function.

    This is what the minimizer calls. Uses jax.value_and_grad for
    efficient simultaneous cost and gradient computation.
    """
    cost_fn = build_cost_fn(
        model, background, observations, B, control_spec,
        template_state, dt, n_steps, checkpoint,
    )
    return jax.value_and_grad(cost_fn)
