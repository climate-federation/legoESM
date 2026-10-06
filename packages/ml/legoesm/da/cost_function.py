"""4D-Var cost functions J(x) = J_b + J_o.

Two cost builders share one rollout core:
- :func:`build_cost_fn` — STATE inversion: the control vector is the initial
  state; recover the initial condition.
- :func:`build_flux_cost_fn` — BOUNDARY / SURFACE-FLUX inversion: the control
  vector is a time-varying surface forcing; the initial state is fixed and the
  forcing that best fits the observations is recovered.

Both are pure functions of the control vector, differentiable by jax.grad, and
share :func:`_obs_cost_rollout` so the adjoint-memory schedules and all
dtype/observation-order/lax.cond guarantees are identical.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable

import jax
import jax.numpy as jnp
from legoesm.da.control_vector import (
    apply_forcing_slice,
    control_to_forcing_series,
    control_to_state,
)
from legoesm.training.checkpoint_schedule import (
    VALID_STORAGE,
    checkpointed_loop,
    offload_policy,
)


def _resolve_and_validate_dispatch(
    checkpoint, checkpoint_schedule, storage, observations, n_steps
):
    """Resolve the effective schedule and validate schedule/storage/obs eagerly.

    Shared by both cost builders so a typo or out-of-window observation fails at
    build time (before tracing), regardless of which inversion is used.
    """
    if checkpoint_schedule is None:
        checkpoint_schedule = "uniform" if checkpoint else "none"
    if checkpoint_schedule not in ("none", "uniform", "binomial"):
        raise ValueError(
            f"unknown checkpoint_schedule {checkpoint_schedule!r}; "
            "expected 'none', 'uniform', or 'binomial'"
        )
    if storage not in VALID_STORAGE:
        raise ValueError(
            f"unknown storage {storage!r}; expected one of {VALID_STORAGE}"
        )
    # Host offload modifies an active checkpoint; "none" has nothing to offload.
    if checkpoint_schedule == "none" and storage != "recompute":
        raise ValueError(
            f"storage {storage!r} requires checkpoint_schedule 'uniform' or "
            "'binomial'; 'none' stores every step and has no checkpoint to offload."
        )
    # Reject out-of-window static observation times up front: the trajectory path
    # would silently clamp an out-of-range gather and the binomial path would
    # silently drop it.  numbers.Integral covers Python int AND numpy integer
    # scalars (np.int64 from data pipelines); a traced/array time_index is dynamic
    # and skipped here (the trajectory path cannot statically validate it either).
    for _obs in observations:
        _ti = _obs.time_index
        if isinstance(_ti, numbers.Integral) and not (-n_steps <= int(_ti) < n_steps):
            raise ValueError(
                f"observation time_index {int(_ti)} out of range for n_steps={n_steps}; "
                "expected within [-n_steps, n_steps)"
            )
    return checkpoint_schedule


def _obs_cost_rollout(
    state_0,
    indexed_step,
    observations,
    n_steps,
    *,
    J_b,
    checkpoint_schedule,
    checkpoints,
    storage,
):
    """Run the forward model accumulating the observation cost J_o.

    ``indexed_step(i, state) -> state`` advances one step given the 0-based traced
    step index ``i`` (so a time-varying forcing can be applied per step).  Shared
    by state inversion and flux inversion: the binomial/none/uniform behaviour and
    every dtype / observation-order / lax.cond guarantee is identical for both.
    """
    jo_dtype = J_b.dtype

    if checkpoint_schedule == "binomial":
        # In-loop observation accumulation: the O(n_steps) trajectory is never
        # materialised, so square-root checkpointing applies.
        def body(carry):
            i, state, slots = carry
            state = indexed_step(i, state)  # state after (i + 1) steps
            new_slots = []
            for obs, slot in zip(observations, slots):
                # Match trajectory[obs.time_index] numpy semantics, incl. the
                # negative-index final-state convention (e.g. -1 -> n_steps-1).
                t = jnp.asarray(obs.time_index)
                t = jnp.where(t < 0, t + n_steps, t)

                def _misfit(s, obs=obs):
                    d = obs.values - obs.operator(s)
                    return (0.5 * jnp.sum(d * (d / (obs.errors ** 2)))).astype(jo_dtype)

                def _zero(s):
                    return jnp.zeros((), dtype=jo_dtype)

                # lax.cond evaluates the operator ONLY at its own timestep — exactly
                # like trajectory[obs.time_index] — so an operator that is non-finite
                # on off-time intermediate states cannot poison the gradient (a
                # jnp.where would still differentiate the dead branch and propagate
                # NaNs).  Each observation accumulates into its OWN slot so the final
                # reduction preserves the caller's observation order (associativity-
                # stable vs the trajectory path for unsorted/disparate-magnitude
                # misfits).
                new_slots.append(slot + jax.lax.cond(i == t, _misfit, _zero, state))
            return (i + 1, state, tuple(new_slots))

        init_slots = tuple(jnp.zeros((), dtype=jo_dtype) for _ in observations)
        _, _, slots = checkpointed_loop(
            body, (jnp.asarray(0), state_0, init_slots), n_steps,
            schedule="binomial", checkpoints=checkpoints, storage=storage,
        )
        # Sum per-observation misfits in the caller's observation order.
        J_o = jnp.zeros((), dtype=jo_dtype)
        for slot in slots:
            J_o = J_o + slot
        return J_o

    # "none" / "uniform": stack the trajectory then index by observation time.
    def scan_step(carry, idx):
        s_new = indexed_step(idx, carry)
        return s_new, s_new

    if checkpoint_schedule == "uniform":
        policy = offload_policy(storage)  # None for recompute, offload for host
        scan_step = (
            jax.checkpoint(scan_step, policy=policy, prevent_cse=False)
            if policy is not None
            else jax.checkpoint(scan_step, prevent_cse=False)
        )

    _, trajectory = jax.lax.scan(scan_step, state_0, jnp.arange(n_steps))

    # Pin each misfit to the control/background dtype so the objective stays in the
    # control's precision and is identical across all schedules.
    J_o = jnp.zeros((), dtype=jo_dtype)
    for obs in observations:
        state_t = jax.tree.map(lambda arr: arr[obs.time_index], trajectory)
        d = obs.values - obs.operator(state_t)
        J_o = J_o + (0.5 * jnp.sum(d * (d / (obs.errors ** 2)))).astype(jo_dtype)
    return J_o


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
    *,
    checkpoint_schedule: str | None = None,
    checkpoints: int | None = None,
    storage: str = "recompute",
) -> Callable[[jax.Array], jax.Array]:
    """Build a JIT-compilable 4D-Var STATE-inversion cost function.

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
    template_state : NamedTuple
        Template state for static fields.
    dt : float
        Model time step [s].
    n_steps : int
        Number of time steps in assimilation window.
    checkpoint : bool
        Back-compat toggle mapping to ``checkpoint_schedule`` when the latter is
        not given: ``True`` -> ``"uniform"``, ``False`` -> ``"none"``.
    checkpoint_schedule : {"none", "uniform", "binomial"}, optional
        Adjoint-memory schedule.  ``"binomial"`` uses Griewank-Walther square-root
        checkpointing (O(sqrt(n_steps)) stored states) — the right choice for long
        windows.  Overrides ``checkpoint`` when set.
    checkpoints : int, optional
        Binomial checkpoint budget K (defaults to ceil(sqrt(n_steps))).
    storage : {"recompute", "host"}
        ``"host"`` offloads matmul residuals to pinned host DRAM (GPU/TPU);
        requires the ``"uniform"`` or ``"binomial"`` schedule.
    """
    checkpoint_schedule = _resolve_and_validate_dispatch(
        checkpoint, checkpoint_schedule, storage, observations, n_steps
    )

    def cost_fn(x: jax.Array) -> jax.Array:
        dx = x - background
        J_b = 0.5 * jnp.sum(dx * B.inv_multiply(dx))
        state_0 = control_to_state(x, control_spec, template_state)

        def indexed_step(i, s):  # state inversion: forcing is fixed in the model
            return model.step(s, dt)

        J_o = _obs_cost_rollout(
            state_0, indexed_step, observations, n_steps,
            J_b=J_b, checkpoint_schedule=checkpoint_schedule,
            checkpoints=checkpoints, storage=storage,
        )
        return J_b + J_o

    return cost_fn


def build_vspace_cost_fn(
    model,
    background: jax.Array,
    observations: tuple,
    B,
    control_spec,
    template_state,
    dt: float,
    n_steps: int,
    checkpoint: bool = True,
    *,
    checkpoint_schedule: str | None = None,
    checkpoints: int | None = None,
    storage: str = "recompute",
) -> Callable[[jax.Array], jax.Array]:
    """4D-Var cost in the control variable v, with x = x_b + B^{1/2} v.

    J(v) = 1/2 v^T v + J_o(x_b + B^{1/2} v)

    For B = U U^T with U invertible this equals the x-space cost of
    :func:`build_cost_fn` at x = x_b + U v; it needs only ``B.sqrt_multiply``,
    so it is also defined where B^{-1} is not (GEN_BE on MPAS meshes, #1819).
    Arguments as in :func:`build_cost_fn`.
    """
    checkpoint_schedule = _resolve_and_validate_dispatch(
        checkpoint, checkpoint_schedule, storage, observations, n_steps
    )

    def cost_fn(v: jax.Array) -> jax.Array:
        J_b = 0.5 * jnp.sum(v * v)
        x = background + B.sqrt_multiply(v)
        state_0 = control_to_state(x, control_spec, template_state)

        def indexed_step(i, s):
            return model.step(s, dt)

        J_o = _obs_cost_rollout(
            state_0, indexed_step, observations, n_steps,
            J_b=J_b, checkpoint_schedule=checkpoint_schedule,
            checkpoints=checkpoints, storage=storage,
        )
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
    *,
    checkpoint_schedule: str | None = None,
    checkpoints: int | None = None,
    storage: str = "recompute",
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    """Build JIT-compiled (J, nabla J) for state inversion (jax.value_and_grad)."""
    cost_fn = build_cost_fn(
        model, background, observations, B, control_spec,
        template_state, dt, n_steps, checkpoint,
        checkpoint_schedule=checkpoint_schedule,
        checkpoints=checkpoints,
        storage=storage,
    )
    return jax.value_and_grad(cost_fn)


def build_flux_cost_fn(
    step_with_forcing,
    initial_state,
    background: jax.Array,
    observations: tuple,
    B,
    forcing_spec,
    template_forcing,
    dt: float,
    n_steps: int,
    checkpoint: bool = True,
    *,
    checkpoint_schedule: str | None = None,
    checkpoints: int | None = None,
    storage: str = "recompute",
) -> Callable[[jax.Array], jax.Array]:
    """Build a JIT-compilable 4D-Var BOUNDARY / SURFACE-FLUX inversion cost.

    The control vector is a TIME-VARYING surface forcing (one slice per step, see
    :func:`legoesm.da.control_vector.build_forcing_control_spec`); the initial
    state is FIXED.  Recovers the forcing (wind stress, heat flux, SST, ...) that
    best fits the observations:

    J(f) = 1/2 (f - f_b)^T B^{-1} (f - f_b)
          + 1/2 sum_i (y_i - H_i(M_i(state0, f)))^T R_i^{-1} (...)

    Parameters
    ----------
    step_with_forcing : callable
        ``step_with_forcing(state, forcing_i, dt) -> state`` — one model step that
        consumes the per-step forcing slice differentiably.  For components whose
        step takes forcing as a separate argument (land, sea ice) this is the step
        directly; for components where forcing lives in the state (ocean) the
        adapter injects ``forcing_i`` into the state before stepping.
    initial_state : pytree
        Fixed background initial condition (NOT optimised).
    background : jax.Array
        Background forcing control vector f_b (first guess).
    observations : tuple of Observation
    B : forcing-error covariance with .inv_multiply (e.g. DiagonalB).
    forcing_spec : ForcingControlSpec
        From ``build_forcing_control_spec``; ``n_window`` must equal ``n_steps``.
    template_forcing : NamedTuple
        Forcing instance providing the uncontrolled leaves + Field metadata.
    dt, n_steps, checkpoint, checkpoint_schedule, checkpoints, storage
        As in :func:`build_cost_fn`.
    """
    if forcing_spec.n_window != n_steps:
        raise ValueError(
            f"forcing_spec.n_window ({forcing_spec.n_window}) must equal "
            f"n_steps ({n_steps}) — one forcing slice per model step."
        )
    background = jnp.asarray(background)
    if background.ndim != 1 or background.shape[0] != forcing_spec.total_size:
        raise ValueError(
            f"background forcing control has shape {tuple(background.shape)}; "
            f"expected a 1-D array of size {forcing_spec.total_size}"
        )
    checkpoint_schedule = _resolve_and_validate_dispatch(
        checkpoint, checkpoint_schedule, storage, observations, n_steps
    )

    def cost_fn(x: jax.Array) -> jax.Array:
        # Validate the candidate control shape BEFORE it enters the prior term so a
        # malformed (e.g. broadcastable 2-D) control cannot corrupt J_b/its gradient.
        series = control_to_forcing_series(x, forcing_spec)
        dx = x - background
        J_b = 0.5 * jnp.sum(dx * B.inv_multiply(dx))

        def indexed_step(i, s):
            forcing_i = apply_forcing_slice(template_forcing, series, i)
            return step_with_forcing(s, forcing_i, dt)

        J_o = _obs_cost_rollout(
            initial_state, indexed_step, observations, n_steps,
            J_b=J_b, checkpoint_schedule=checkpoint_schedule,
            checkpoints=checkpoints, storage=storage,
        )
        return J_b + J_o

    return cost_fn


def build_flux_cost_and_grad_fn(
    step_with_forcing,
    initial_state,
    background: jax.Array,
    observations: tuple,
    B,
    forcing_spec,
    template_forcing,
    dt: float,
    n_steps: int,
    checkpoint: bool = True,
    *,
    checkpoint_schedule: str | None = None,
    checkpoints: int | None = None,
    storage: str = "recompute",
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    """Build JIT-compiled (J, nabla J) for flux inversion (jax.value_and_grad)."""
    cost_fn = build_flux_cost_fn(
        step_with_forcing, initial_state, background, observations, B,
        forcing_spec, template_forcing, dt, n_steps, checkpoint,
        checkpoint_schedule=checkpoint_schedule,
        checkpoints=checkpoints,
        storage=storage,
    )
    return jax.value_and_grad(cost_fn)
