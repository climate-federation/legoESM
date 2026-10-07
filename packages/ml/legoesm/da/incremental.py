"""Incremental 4D-Var (Courtier, Thepaut & Hollingsworth 1994).

Outer loop linearizes around a trajectory; inner loop minimizes a
quadratic approximation. The adjoint is computed automatically by
jax.grad through the inner loop cost function.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.da.control_vector import (
    ControlVectorSpec,
    control_to_state,
    state_to_control,
)
from legoesm.da.cost_function import build_cost_fn, build_vspace_cost_fn
from legoesm.da.minimizer import minimize_cg, minimize_lbfgs
from legoesm.da.preconditioning import preconditioned_cost_fn

logger = logging.getLogger(__name__)


def _uses_vspace_background(B) -> bool:
    """GEN_BE covariances are minimised in the control variable v (J_b = 1/2|v|^2).

    Exact wherever GEN_BE has an inverse, and the only option where it has none
    (MPAS meshes, #1819).
    """
    from legoesm.da.gen_be import GenBETransform

    return isinstance(B, GenBETransform)


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
    # ||grad J|| at each outer start.  In v (x = x_b + B^{1/2} v) for a
    # GenBETransform with use_preconditioning=True, otherwise in x; the outer
    # early exit compares this same norm with inner_gtol.
    grad_norm_history: list
    inner_iterations: list
    # RMS of y - H(M(x)) over all obs, after each outer iteration; NaN when
    # there are no observation values (the RMS of an empty set is undefined).
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
        Must have .sqrt_multiply(); also .inv_multiply() unless it is a
        GenBETransform with preconditioning on, whose cost is taken in the
        control variable (J_b = 1/2|v|^2; the reported gradient is then with
        respect to v).
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
    vspace = _uses_vspace_background(B)
    if vspace and not config.use_preconditioning:
        raise ValueError(
            "incremental_4dvar: GenBETransform needs use_preconditioning=True. The "
            "x-space cost (use_preconditioning=False) needs B^{-1}, which GenBE does "
            "not provide on MPAS meshes and provides only on the subspace U^{-1} U "
            "preserves on the Gaussian grid, so its analysis is not the B = U U^T "
            "one (#1819). Fix: IncrementalConfig(use_preconditioning=True)."
        )
    x_b = state_to_control(background_state, control_spec)
    x_k = x_b.copy()
    # Preconditioned iterate, x_k = x_b + B^{1/2} v_k; carried across outer
    # iterations so each inner solve warm-starts from the current analysis.
    v_k = jnp.zeros_like(x_b)

    cost_history = []
    grad_norm_history = []
    inner_iterations = []
    innovation_rms_list = []

    # Build the cost-and-grad ONCE with the re-linearization template as a
    # TRACED argument.  Each outer iteration rebinds the template VALUE (same
    # shape), so the compiled function is REUSED instead of re-JITed every outer
    # loop — ``build_cost_fn`` runs inside the trace and compiles once.
    # (Previously the cost closure was rebuilt and ``jax.jit``-wrapped INSIDE the
    # loop, recompiling the full windowed model + adjoint on every outer
    # iteration.)  Mirrors the SegmentForcing doctrine: per-iteration changing
    # values are traced args, not compile-time closure captures.
    @jax.jit
    def _cost_and_grad(x, template_state):
        cost_fn = build_cost_fn(
            model, x_b, observations, B, control_spec,
            template_state, dt, n_steps, config.checkpoint,
        )
        return jax.value_and_grad(cost_fn)(x)

    @jax.jit
    def _precond_and_grad(v, template_state):
        cost_fn = build_cost_fn(
            model, x_b, observations, B, control_spec,
            template_state, dt, n_steps, config.checkpoint,
        )
        J_tilde = preconditioned_cost_fn(cost_fn, B, x_b)
        return jax.value_and_grad(J_tilde)(v)

    @jax.jit
    def _vspace_and_grad(v, template_state):
        cost_fn = build_vspace_cost_fn(
            model, x_b, observations, B, control_spec,
            template_state, dt, n_steps, config.checkpoint,
        )
        return jax.value_and_grad(cost_fn)(v)

    use_v = vspace and config.use_preconditioning
    precond_and_grad = _vspace_and_grad if use_v else _precond_and_grad

    n_obs_values = sum(int(jnp.size(o.values)) for o in observations)

    @jax.jit
    def _innovation_rms(x):
        # Forward-only rollout accumulating sum (y - H(M(x)))^2 in-loop (no
        # stored trajectory); obs time convention matches the cost rollout.
        def body(s, i):
            s = model.step(s, dt)
            sq = jnp.zeros((), x.dtype)
            for o in observations:
                t = jnp.asarray(o.time_index)
                t = jnp.where(t < 0, t + n_steps, t)
                sq = sq + jax.lax.cond(
                    i == t,
                    lambda s, o=o: jnp.sum((o.values - o.operator(s)) ** 2).astype(x.dtype),
                    lambda s: jnp.zeros((), x.dtype),
                    s)
            return s, sq

        state = control_to_state(x, control_spec, background_state)
        _, sq = jax.lax.scan(body, state, jnp.arange(n_steps))
        return jnp.sqrt(jnp.sum(sq) / n_obs_values)

    for outer in range(config.n_outer):
        # Re-linearize around the current iterate (changing VALUE, fixed shape).
        template = control_to_state(x_k, control_spec, background_state)

        # Current cost and gradient (compiled once; no per-outer recompile).
        if use_v:
            J_k, g_k = _vspace_and_grad(v_k, template)
        else:
            J_k, g_k = _cost_and_grad(x_k, template)
        g_norm = float(jnp.linalg.norm(g_k))
        cost_history.append(float(J_k))
        grad_norm_history.append(g_norm)

        logger.info(
            f"Outer {outer}: J={float(J_k):.6e}, ||grad||={g_norm:.6e}"
        )

        if g_norm < config.inner_gtol:
            inner_iterations.append(0)
            innovation_rms_list.append(float(_innovation_rms(x_k)))
            continue

        # Inner loop minimization.  ``partial`` binds the current template by
        # VALUE (no late-binding / B023), giving the minimizer a single-arg
        # ``f(x) -> (J, grad)`` backed by the once-compiled wrapper above.
        if config.use_preconditioning:
            v0 = v_k
            inner_fn = partial(precond_and_grad, template_state=template)

            if config.inner_method == "cg":
                result = minimize_cg(
                    inner_fn, v0,
                    max_iter=config.n_inner, gtol=config.inner_gtol,
                )
            else:
                result = minimize_lbfgs(
                    inner_fn, v0,
                    max_iter=config.n_inner, gtol=config.inner_gtol,
                )

            v_k = result.x
            x_k = x_b + B.sqrt_multiply(v_k)
        else:
            inner_fn = partial(_cost_and_grad, template_state=template)
            if config.inner_method == "cg":
                result = minimize_cg(
                    inner_fn, x_k,
                    max_iter=config.n_inner, gtol=config.inner_gtol,
                )
            else:
                result = minimize_lbfgs(
                    inner_fn, x_k,
                    max_iter=config.n_inner, gtol=config.inner_gtol,
                )
            x_k = result.x

        inner_iterations.append(int(result.n_iter))

        innovation_rms_list.append(float(_innovation_rms(x_k)))

    analysis_state = control_to_state(x_k, control_spec, background_state)

    diagnostics = IncrementalDiagnostics(
        cost_history=cost_history,
        grad_norm_history=grad_norm_history,
        inner_iterations=inner_iterations,
        innovation_rms=innovation_rms_list,
    )

    return analysis_state, diagnostics
