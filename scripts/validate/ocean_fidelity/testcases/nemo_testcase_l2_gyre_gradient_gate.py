#!/usr/bin/env python3
"""Reverse/forward/FD and adjoint-resource gate for certified GYRE-zco.

The numerical protocol is frozen in
``nemo_testcases_l2_gyre_gradgate_round1_prereg.md``.  This module deliberately
contains no physics repair: a poisoned adjoint is an exit-1 measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
from legoesm.ocean.vertical import (
    nemo_qco_live_face_geometry_cgrid,
    nemo_qco_live_t_thickness,
    nemo_qco_resolved_mesh_operands,
)
from legoesm.training.checkpoint_schedule import checkpointed_loop, sqrt_checkpoints

CASE = "GYRE-zco"
DT_S = 14400.0
REVFWD_TOL = 1.0e-9
FD_TOL = 0.05
FD_STEP = 1.0e-3
PERTURBATION_K = 1.0e-3
SCHEDULE_RTOL = 1.0e-9
SCHEDULE_ATOL = 1.0e-11
SCHEDULES = ("none", "uniform", "binomial")
AGREEMENT_COMPOSITIONS = ("per_step", *SCHEDULES)
AGREEMENT_STEPS = (1, 6, 60, 360)
RESOURCE_STEPS = (1, 6, 60, 360)
FIELDS = ("T", "S", "u", "v")
GPU_NATIVE_POLICY_LABEL = (
    "GPU, XLA transcendentals — not the certified bit-exact policy"
)
CPU_LIBM_POLICY_LABEL = "CPU, certified fp64/libm bit-exact policy"
SCHEDULE_SCOPE = (
    "cold production step followed by a component-agnostic checkpoint "
    "schedule over warm production jitted steps"
)
RESOURCE_SCHEMA = "gyre-gradgate-resource-v3-primal-preserving-boundary"
REVERSE_TRANSFORM = "jax.jit(jax.value_and_grad(scalar_line))"
WARM_TIMER_SPAN = (
    "blocked execution of pre-jitted value_and_grad only; sampler start and "
    "stop/join excluded"
)


class ScoredState(NamedTuple):
    T: jax.Array
    S: jax.Array
    u: jax.Array
    v: jax.Array
    eta: jax.Array


class LossReference(NamedTuple):
    state: ScoredState
    T_volume: jax.Array
    S_volume: jax.Array
    u_volume: jax.Array
    v_volume: jax.Array
    T_variance: jax.Array
    S_variance: jax.Array
    u_variance: jax.Array
    v_variance: jax.Array


class GYREContext(NamedTuple):
    card: object
    model: LatLonCGridOceanModel
    initial_state: object
    surface_forcings: object
    observed_ad_step: object
    checkpointed_step: object


class ReversePassMeasurement(NamedTuple):
    value: jax.Array
    gradient: jax.Array
    wall_seconds: float
    sampler: object


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def measurement_policy_label(backend: str, transcendentals: str) -> str:
    """Return a fail-closed human label for a measured execution policy."""
    if backend == "gpu" and transcendentals == "native":
        return GPU_NATIVE_POLICY_LABEL
    if backend == "cpu" and transcendentals == "libm":
        return CPU_LIBM_POLICY_LABEL
    raise ValueError(
        "unsupported GYRE gradient measurement policy: "
        f"backend={backend!r}, transcendentals={transcendentals!r}"
    )


def runtime_identity() -> dict:
    """Return the logical device and fail-closed physical GPU selection."""
    backend = jax.default_backend()
    identity = {
        "backend": backend,
        "device": str(jax.devices()[0]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_index": None,
    }
    if backend == "gpu":
        require(
            identity["cuda_visible_devices"] == "1",
            "GPU measurements require CUDA_VISIBLE_DEVICES=1",
        )
        identity["physical_gpu_index"] = 1
    return identity


def configure_runtime(transcendentals: str = "libm") -> None:
    """Select and verify the requested fp64 transcendental policy."""
    require(bool(jax.config.jax_enable_x64), "JAX_ENABLE_X64=1 is required")
    policy = PrecisionPolicy.fp64(transcendentals=transcendentals)
    set_policy(policy)
    require(
        get_policy() == policy,
        f"precision policy is not {policy!r}",
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_surface_forcings():
    """Reuse the certified campaign forcing builder without copying it."""
    path = Path(__file__).with_name("nemo_testcase_l2_gyre_phase3_gate.py")
    spec = importlib.util.spec_from_file_location("gyre_phase3_gradient_forcing", path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._surface_forcings


@jax.custom_jvp
def primal_preserving_boundary(production, replay):
    """Return ``production`` while differentiating only through ``replay``.

    Both arguments are equal-structure pytrees.  No value is detached: the
    custom rule explicitly chooses the replay tangent, whose transpose is the
    replay pullback, while retaining every bit of the separately evaluated
    production primal.
    """
    return production


@primal_preserving_boundary.defjvp
def _primal_preserving_boundary_jvp(primals, tangents):
    production, _replay = primals
    _production_tangent, replay_tangent = tangents
    return production, replay_tangent


def build_primal_preserving_step(
        production_step, replay_step, *, plant_replay_primal=False):
    """Build a step with distinct production-primal and replay-AD arms."""

    observed_step = build_observed_primal_preserving_step(
        production_step, replay_step,
        plant_replay_primal=plant_replay_primal)

    def bounded_step(*args):
        return observed_step(*args)[0]

    return bounded_step


def build_observed_primal_preserving_step(
        production_step, replay_step, *, plant_replay_primal=False):
    """Expose the raw transform replay without composing it as the primal."""

    @jax.custom_jvp
    def observed_step(*args):
        production = production_step(*args)
        replay = replay_step(*args)
        return production, replay

    @observed_step.defjvp
    def observed_step_jvp(primals, tangents):
        # This direct call receives the rule's primal values, so it executes
        # the ordinary production program before any upstream JVP/VJP can
        # replace its primal with the transform replay.
        production = production_step(*primals)
        replay, replay_tangent = jax.jvp(
            replay_step, primals, tangents)
        selected = (
            replay if plant_replay_primal
            else primal_preserving_boundary(production, replay)
        )
        return (selected, replay), (replay_tangent, replay_tangent)

    return observed_step


def build_context(*, transcendentals: str = "libm") -> GYREContext:
    """Build the certified card and prime only production step caches."""
    configure_runtime(transcendentals)
    card = build_nemo_testcase_card(CASE)
    require(card.dt_s == DT_S, f"card dt changed: {card.dt_s}")
    cfg = card.recipe.model_config
    require(cfg.outer_integrator == "forward_euler", "GYRE outer dispatch changed")
    require(cfg.barotropic.n_barotropic_substeps == 50, "GYRE substeps changed")
    require(cfg.physics.vertical_mixing.scheme == "tke", "GYRE TKE changed")
    require(
        cfg.physics.vertical_mixing.tke.tke_matrix_evaluation == "nemo_literal",
        "GYRE TKE matrix is not NEMO-literal",
    )
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    initial = model._seed_tke_preclosure_carry(card.recipe.initial_state)
    model.prime_step_caches(initial)
    require(initial.T.data.dtype == jnp.float64, "GYRE state is not float64")
    require(card.recipe.z_coord.dz_ref.dtype == jnp.float64,
            "GYRE geometry is not float64")
    surface_forcings = _load_surface_forcings()

    def production_step(state, kt):
        freshwater, surface = surface_forcings(card, state, kt)
        return model._step_jitted(state, card.dt_s, freshwater, surface)

    # Round-9 user decision: the local AD boundary returns a separately
    # executed production primal while the unchanged step program supplies its
    # JVP/pullback. Decision 49 still rematerializes the whole bounded step in
    # the per-step composition; checkpoint schedules wrap ``ad_step`` later.
    observed_ad_step = jax.jit(build_observed_primal_preserving_step(
        production_step, production_step))
    checkpointed_step = jax.jit(jax.checkpoint(
        lambda state, kt: observed_ad_step(state, kt)[0]))
    return GYREContext(
        card, model, initial, surface_forcings, observed_ad_step,
        checkpointed_step)


def _score_state(state) -> ScoredState:
    return ScoredState(
        state.T.data, state.S.data, state.u.data, state.v.data, state.eta.data)


def _apply_initial_perturbation(context: GYREContext, d_t0):
    return context.initial_state._replace(
        T=context.initial_state.T.replace(
            data=context.initial_state.T.data + d_t0))


def _production_step(context: GYREContext, state, kt: int):
    freshwater, surface = context.surface_forcings(context.card, state, kt)
    return context.model._step_jitted(
        state, context.card.dt_s, freshwater, surface)


def _ad_production_step(context: GYREContext, state, kt: int, *, remat=False):
    """Run one production step through the approved local AD boundary."""
    if remat:
        return context.checkpointed_step(state, kt)
    return context.observed_ad_step(state, kt)[0]


def production_rollout(context: GYREContext, d_t0, n_steps: int):
    """Campaign-order Python loop; every step calls production ``_step_jitted``."""
    state = _apply_initial_perturbation(context, d_t0)
    for kt in range(1, n_steps + 1):
        state = _production_step(context, state, kt)
    return state


def gradient_rollout(context: GYREContext, d_t0, n_steps: int):
    """Twin-Decision-49 Python composition of rematerialized production steps."""
    state = _apply_initial_perturbation(context, d_t0)
    for kt in range(1, n_steps + 1):
        state = _ad_production_step(context, state, kt, remat=True)
    return state


def scheduled_rollout(context: GYREContext, d_t0, n_steps: int, schedule: str):
    """Segment the pytree-changing cold step from the warm scheduled loop."""
    if schedule == "per_step":
        return gradient_rollout(context, d_t0, n_steps)
    state = _apply_initial_perturbation(context, d_t0)
    if n_steps <= 0:
        return state

    # GYRE's first production step materializes TKE, barotropic-history, and
    # RK3 carry leaves.  A scan cannot change its carry treedef, so the cold
    # step is its own adjoint segment and only the fixed-pytree warm loop is
    # handed to the component-agnostic checkpoint scheduler.
    state = _ad_production_step(context, state, 1)
    if n_steps == 1:
        return state

    def body(carry):
        kt, step_state = carry
        next_kt = kt + 1
        return next_kt, _ad_production_step(context, step_state, next_kt)

    _, state = checkpointed_loop(
        body,
        (jnp.asarray(1, dtype=jnp.int32), state),
        n_steps - 1,
        schedule=schedule,
    )
    return state


def _live_volumes(context: GYREContext, truth: ScoredState):
    """Twin-branch live T/U/V control volumes at the scored after-state."""
    card = context.card
    state = context.initial_state
    grid = card.recipe.grid
    z_coord = card.recipe.z_coord
    dtype = truth.T.dtype
    u_mask_3d, v_mask_3d = compute_face_masks_3d(z_coord.is_active, grid)
    operands = nemo_qco_resolved_mesh_operands(
        z_coord, grid, u_mask_3d, v_mask_3d, dtype, truth.T.shape[-1])
    active_t = (
        jnp.asarray(z_coord.is_active, dtype=dtype)
        * jnp.asarray(state.land_mask.data[..., None], dtype=dtype)
    )
    active_u = jnp.asarray(u_mask_3d, dtype=dtype)
    active_v = jnp.asarray(v_mask_3d, dtype=dtype)
    area_t = jnp.asarray(operands.area_t, dtype=dtype)
    area_u = jnp.asarray(grid.dx_u * grid.dy_u, dtype=dtype)
    area_v = jnp.asarray(grid.dx_v * grid.dy_v, dtype=dtype)
    e3t = nemo_qco_live_t_thickness(
        truth.eta,
        state.H_bathy.data,
        z_coord,
        dtype,
        e3t_0=operands.e3t_0,
    )
    e3u, e3v, _, _ = nemo_qco_live_face_geometry_cgrid(
        truth.eta,
        operands.e3u_0,
        operands.e3v_0,
        operands.umask3,
        operands.vmask3,
        operands.hu_0,
        operands.hv_0,
        operands.area_t,
        operands.area_u,
        operands.area_v,
    )
    return (
        e3t * area_t[..., None] * active_t,
        e3u * area_u[..., None] * active_u,
        e3v * area_v[..., None] * active_v,
    )


def volume_weighted_mse(pred, target, volume):
    """The gradient-twin weighted squared-error reducer."""
    volume = jnp.asarray(volume, dtype=pred.dtype)
    denominator = jnp.sum(volume)
    return jnp.sum(volume * (pred - target) ** 2) / denominator


def build_reference(context: GYREContext, truth_state) -> LossReference:
    """Freeze the after-state truth, live volumes, and four normalizers."""
    truth = _score_state(truth_state)
    t_volume, u_volume, v_volume = _live_volumes(context, truth)
    volumes = {"T": t_volume, "S": t_volume, "u": u_volume, "v": v_volume}
    variances = {}
    for name in FIELDS:
        field = getattr(truth, name)
        weight = volumes[name]
        mean = jnp.sum(field * weight) / jnp.sum(weight)
        variance = volume_weighted_mse(field, mean, weight)
        value = float(np.asarray(variance))
        require(math.isfinite(value) and value > 0.0,
                f"{name} truth variance is not positive and finite: {value}")
        variances[name] = variance
    return LossReference(
        truth,
        t_volume,
        t_volume,
        u_volume,
        v_volume,
        variances["T"],
        variances["S"],
        variances["u"],
        variances["v"],
    )


def make_loss(context: GYREContext, reference: LossReference, n_steps: int,
              *, schedule: str | None = None):
    """Return the registered four-field normalized after-state loss."""

    def loss(d_t0):
        state = (
            gradient_rollout(context, d_t0, n_steps)
            if schedule is None
            else scheduled_rollout(context, d_t0, n_steps, schedule)
        )
        candidate = _score_state(state)
        scores = []
        for name in FIELDS:
            scores.append(
                volume_weighted_mse(
                    getattr(candidate, name),
                    getattr(reference.state, name),
                    getattr(reference, f"{name}_volume"),
                )
                / getattr(reference, f"{name}_variance")
            )
        return jnp.mean(jnp.stack(scores))

    return loss


def loss_at_state(reference: LossReference, state):
    """Evaluate the registered objective from an already-produced after-state."""
    candidate = _score_state(state)
    scores = [
        volume_weighted_mse(
            getattr(candidate, name),
            getattr(reference.state, name),
            getattr(reference, f"{name}_volume"),
        ) / getattr(reference, f"{name}_variance")
        for name in FIELDS
    ]
    return jnp.mean(jnp.stack(scores))


def build_compiled_step_transforms(context: GYREContext):
    """Build reusable one-production-step VJP and JVP executables.

    ``kt`` is a dynamic argument, so a multi-step rollout does not create one
    transform closure per forcing time.  JAX specializes these functions only
    for the cold and warm state pytrees.
    """

    observed_step = context.observed_ad_step

    def step_vjp(state, kt, cotangent):
        (selected, replay), pullback = jax.vjp(
            lambda step_state: observed_step(step_state, kt),
            state,
        )
        # Mirror the already-valid output cotangent tree so integer/bool leaves
        # retain JAX's float0 tangent dtype instead of receiving integer zeros.
        zero_replay_cotangent = jax.tree_util.tree_map(
            jnp.zeros_like, cotangent)
        state_cotangent = pullback(
            (cotangent, zero_replay_cotangent))[0]
        return selected, replay, state_cotangent

    def step_jvp(state, kt, tangent):
        (selected, replay), (selected_tangent, _replay_tangent) = jax.jvp(
            lambda step_state: observed_step(step_state, kt),
            (state,),
            (tangent,),
        )
        return selected, replay, selected_tangent

    return jax.jit(step_vjp), jax.jit(step_jvp)


def step_transform_footprint(n_steps: int) -> dict:
    """Describe the structure-specialized programs used by explicit AD."""
    require(n_steps >= 1, "agreement requires at least one production step")
    warm_steps = max(0, n_steps - 1)
    return {
        "whole_window_transform_jitted": False,
        "distinct_state_pytree_specializations": 1 + int(warm_steps > 0),
        "cold_program_invocations": 1,
        "warm_program_invocations": warm_steps,
        "vjp_step_invocations": n_steps,
        "jvp_step_invocations": n_steps,
        "dynamic_model_step_argument": True,
        "production_step_traces_per_specialization": 2,
    }


def enforce_step_replay_byte_gate(
        row: dict, vjp_boundary_reports: dict, jvp_boundary_reports: dict,
        vjp_replay_reports: dict, jvp_replay_reports: dict) -> dict:
    """Require boundary outputs to preserve every production-state bit."""
    vjp_boundary_bitwise = all(
        report["bitwise_equal"] for report in vjp_boundary_reports.values())
    jvp_boundary_bitwise = all(
        report["bitwise_equal"] for report in jvp_boundary_reports.values())
    vjp_replay_bitwise = all(
        report["bitwise_equal"] for report in vjp_replay_reports.values())
    jvp_replay_bitwise = all(
        report["bitwise_equal"] for report in jvp_replay_reports.values())
    diagnostics = {
        "step_vjp_boundary_reports": vjp_boundary_reports,
        "step_vjp_boundary_outputs_bitwise": vjp_boundary_bitwise,
        "step_jvp_boundary_reports": jvp_boundary_reports,
        "step_jvp_boundary_outputs_bitwise": jvp_boundary_bitwise,
        "step_vjp_replay_reports": vjp_replay_reports,
        "step_vjp_replay_used_as_primal": False,
        "step_vjp_replays_bitwise": vjp_replay_bitwise,
        "step_jvp_replay_reports": jvp_replay_reports,
        "step_jvp_replay_used_as_primal": False,
        "step_jvp_replays_bitwise": jvp_replay_bitwise,
    }
    if vjp_boundary_bitwise and jvp_boundary_bitwise:
        row.update(diagnostics)
        return row
    refused = unmeasured_agreement(
        "production-step local AD boundary changed the production primal")
    refused["refused_derivative_evidence"] = {
        key: row.get(key)
        for key in (
            "status",
            "reverse",
            "forward",
            "finite_difference",
            "rev_fwd_relative_error",
            "rev_fd_relative_error",
            "gradient_dot_abs_sum",
            "gradient_direction_norm_product",
            "rev_fwd_absolute_discrepancy",
            "rev_fwd_scale_aware_error",
        )
    }
    refused.update(diagnostics)
    return refused


def explicit_stepwise_directional_derivatives(
        context: GYREContext, reference: LossReference, base, direction,
        n_steps: int, *, plant: bool = False):
    """Compose one VJP/JVP per production step without a whole-window trace."""
    states = [_apply_initial_perturbation(context, base)]
    for kt in range(1, n_steps + 1):
        states.append(_production_step(context, states[-1], kt))
        jax.block_until_ready(states[-1])

    value, objective_pullback = jax.vjp(
        lambda state: loss_at_state(reference, state), states[-1])
    cotangent = objective_pullback(jnp.ones_like(value))[0]
    compiled_step_vjp, compiled_step_jvp = build_compiled_step_transforms(context)
    boundary_reports = {}
    replay_reports = {}
    for kt in range(n_steps, 0, -1):
        previous = states[kt - 1]
        selected, replay, cotangent = compiled_step_vjp(
            previous,
            jnp.asarray(kt, dtype=jnp.int32),
            cotangent,
        )
        jax.block_until_ready(selected)
        boundary_reports[str(kt)] = state_bitwise_report(
            states[kt], selected)
        jax.block_until_ready(replay)
        replay_reports[str(kt)] = state_bitwise_report(states[kt], replay)
        jax.block_until_ready(cotangent)
    gradient = cotangent.T.data
    if plant:
        _, plant_pullback = jax.vjp(broken_vjp_identity, base)
        gradient = plant_pullback(gradient)[0]
    reverse = float(np.asarray(jnp.vdot(gradient, direction)))

    zero = jnp.asarray(0.0, dtype=base.dtype)
    _, tangent = jax.jvp(
        lambda alpha: _apply_initial_perturbation(
            context, base + alpha * direction),
        (zero,),
        (jnp.asarray(1.0, dtype=base.dtype),),
    )
    forward_boundary_reports = {}
    forward_replay_reports = {}
    for kt in range(1, n_steps + 1):
        selected, replay, tangent = compiled_step_jvp(
            states[kt - 1],
            jnp.asarray(kt, dtype=jnp.int32),
            tangent,
        )
        jax.block_until_ready(selected)
        forward_boundary_reports[str(kt)] = state_bitwise_report(
            states[kt], selected)
        jax.block_until_ready(replay)
        forward_replay_reports[str(kt)] = state_bitwise_report(
            states[kt], replay)
    _, forward_tangent = jax.jvp(
        lambda after: loss_at_state(reference, after),
        (states[-1],),
        (tangent,),
    )
    forward = float(np.asarray(forward_tangent))

    def finite_difference_loss(alpha):
        after = production_rollout(
            context, base + alpha * direction, n_steps)
        return loss_at_state(reference, after)

    plus = float(np.asarray(finite_difference_loss(FD_STEP)))
    minus = float(np.asarray(finite_difference_loss(-FD_STEP)))
    finite_difference = (plus - minus) / (2.0 * FD_STEP)
    row = classify_agreement(reverse, forward, finite_difference)
    dot_abs_sum = float(np.asarray(jnp.sum(jnp.abs(gradient * direction))))
    norm_product = float(np.asarray(
        jnp.linalg.norm(gradient) * jnp.linalg.norm(direction)))
    discrepancy = abs(reverse - forward)
    row.update({
        "gradient_dot_abs_sum": dot_abs_sum,
        "gradient_direction_norm_product": norm_product,
        "rev_fwd_absolute_discrepancy": discrepancy,
        "rev_fwd_scale_aware_error": (
            discrepancy / max(dot_abs_sum, 1.0e-300)),
        "step_transform_compile_footprint": step_transform_footprint(n_steps),
    })
    return enforce_step_replay_byte_gate(
        row, boundary_reports, forward_boundary_reports,
        replay_reports, forward_replay_reports)


def perturbations(context: GYREContext):
    """Return the registered evaluation point and independent unit direction."""
    wet = (
        jnp.asarray(context.card.recipe.z_coord.is_active, dtype=jnp.float64)
        * context.initial_state.land_mask.data[..., None]
    )
    base = jax.random.normal(
        jax.random.PRNGKey(0), context.initial_state.T.data.shape,
        dtype=jnp.float64) * wet
    direction = jax.random.normal(
        jax.random.PRNGKey(42), context.initial_state.T.data.shape,
        dtype=jnp.float64) * wet
    base = PERTURBATION_K * base / jnp.linalg.norm(base)
    direction = direction / jnp.linalg.norm(direction)
    return base, direction


@jax.custom_vjp
def broken_vjp_identity(value):
    """Primal identity whose planted pullback doubles every cotangent."""
    return value


def _broken_vjp_fwd(value):
    return value, None


def _broken_vjp_bwd(_residual, cotangent):
    return (2.0 * cotangent,)


broken_vjp_identity.defvjp(_broken_vjp_fwd, _broken_vjp_bwd)


def classify_agreement(reverse: float, forward: float, finite_difference: float):
    """Fail-closed classification shared by the real row and plant test."""
    finite = all(math.isfinite(value)
                 for value in (reverse, forward, finite_difference))
    nonzero = finite and all(
        value != 0.0 for value in (reverse, forward, finite_difference))
    revfwd = abs(reverse - forward) / max(
        abs(reverse), abs(forward), 1.0e-300)
    revfd = abs(reverse - finite_difference) / max(
        abs(reverse), abs(finite_difference), 1.0e-300)
    passed = finite and nonzero and revfwd <= REVFWD_TOL and revfd <= FD_TOL
    return {
        "status": "CONFIRMED" if passed else "REFUTED",
        "reverse": reverse if math.isfinite(reverse) else None,
        "forward": forward if math.isfinite(forward) else None,
        "finite_difference": (
            finite_difference if math.isfinite(finite_difference) else None),
        "rev_fwd_relative_error": revfwd if math.isfinite(revfwd) else None,
        "rev_fd_relative_error": revfd if math.isfinite(revfd) else None,
        "rev_fwd_bar": REVFWD_TOL,
        "rev_fd_bar": FD_TOL,
        "finite": finite,
        "nonzero": nonzero,
        "gradient_dot_abs_sum": None,
        "gradient_direction_norm_product": None,
        "rev_fwd_absolute_discrepancy": (
            abs(reverse - forward) if finite else None),
        "rev_fwd_scale_aware_error": None,
    }


def unmeasured_agreement(reason: str | None = None) -> dict:
    """Represent a row refused before any derivative value exists."""
    row = {
        "status": "UNMEASURED",
        "reverse": None,
        "forward": None,
        "finite_difference": None,
        "rev_fwd_relative_error": None,
        "rev_fd_relative_error": None,
        "rev_fwd_bar": REVFWD_TOL,
        "rev_fd_bar": FD_TOL,
        "finite": None,
        "nonzero": None,
        "gradient_dot_abs_sum": None,
        "gradient_direction_norm_product": None,
        "rev_fwd_absolute_discrepancy": None,
        "rev_fwd_scale_aware_error": None,
    }
    if reason is not None:
        row["error"] = reason
    return row


def directional_derivatives(
        forward_loss, base, direction, *, reverse_loss=None, plant=False):
    """Measure one direction and its full-gradient cancellation scales.

    ``reverse_loss`` permits a reverse-only checkpoint schedule to be checked
    against the bit-identical forward-mode-capable composition.
    """
    zero = jnp.asarray(0.0, dtype=base.dtype)
    reverse_loss = forward_loss if reverse_loss is None else reverse_loss

    def forward_line(alpha):
        return forward_loss(base + alpha * direction)

    def reverse_objective(value):
        reverse_input = broken_vjp_identity(value) if plant else value
        return reverse_loss(reverse_input)

    gradient = jax.grad(reverse_objective)(base)
    reverse = float(np.asarray(jnp.vdot(gradient, direction)))
    _, forward_value = jax.jvp(
        forward_line, (zero,), (jnp.asarray(1.0, base.dtype),))
    forward = float(np.asarray(forward_value))
    plus = float(np.asarray(
        forward_line(jnp.asarray(FD_STEP, base.dtype))))
    minus = float(np.asarray(
        forward_line(jnp.asarray(-FD_STEP, base.dtype))))
    finite_difference = (plus - minus) / (2.0 * FD_STEP)
    row = classify_agreement(reverse, forward, finite_difference)
    dot_abs_sum = float(np.asarray(jnp.sum(jnp.abs(gradient * direction))))
    norm_product = float(np.asarray(
        jnp.linalg.norm(gradient) * jnp.linalg.norm(direction)))
    absolute_discrepancy = abs(reverse - forward)
    row.update({
        "gradient_dot_abs_sum": dot_abs_sum,
        "gradient_direction_norm_product": norm_product,
        "rev_fwd_absolute_discrepancy": absolute_discrepancy,
        "rev_fwd_scale_aware_error": (
            absolute_discrepancy / max(dot_abs_sum, 1.0e-300)),
    })
    return row


def measure_agreement(n_steps: int, schedule: str, *, plant=False,
                      transcendentals: str = "libm"):
    """Build and evaluate one certified-card agreement row."""
    context = build_context(transcendentals=transcendentals)
    zero = jnp.zeros_like(context.initial_state.T.data)
    truth_state = production_rollout(context, zero, n_steps)
    jax.block_until_ready(truth_state)
    # Agreement follows the SWG lane's accepted composition: a Python loop
    # around the rematerialized production JIT.  The checkpoint schedule name
    # is retained for matrix compatibility but does not replace this byte-exact
    # agreement path with a scan.
    gradient_truth = gradient_rollout(context, zero, n_steps)
    jax.block_until_ready(gradient_truth)
    primal_report = state_bitwise_report(truth_state, gradient_truth)
    forward_schedule = "per_step"
    forward_report = primal_report
    row = unmeasured_agreement()
    evaluation_reports = {}
    if not primal_report["bitwise_equal"]:
        row = unmeasured_agreement(
            f"{schedule} segmented schedule primal differs bitwise from "
            "the production Python loop")
    elif not forward_report["bitwise_equal"]:
        row = unmeasured_agreement(
            "uniform forward-reference primal differs bitwise from the "
            "production Python loop")
    else:
        reference = build_reference(context, truth_state)
        base, direction = perturbations(context)
        perturbations_to_check = {
            "base": base,
            "fd_plus": base + FD_STEP * direction,
            "fd_minus": base - FD_STEP * direction,
        }
        for label, perturbation in perturbations_to_check.items():
            production = production_rollout(context, perturbation, n_steps)
            jax.block_until_ready(production)
            selected = gradient_rollout(context, perturbation, n_steps)
            jax.block_until_ready(selected)
            evaluation_reports[f"per_step_{label}"] = (
                state_bitwise_report(production, selected))
        bad_evaluation_reports = [
            name for name, report in evaluation_reports.items()
            if not report["bitwise_equal"]
        ]
        if bad_evaluation_reports:
            row = unmeasured_agreement(
                "scheduled primal differs bitwise from production at "
                + ", ".join(bad_evaluation_reports))
        else:
            try:
                row = explicit_stepwise_directional_derivatives(
                    context, reference, base, direction, n_steps,
                    plant=plant)
            except (TypeError, ValueError, RuntimeError) as error:
                row = unmeasured_agreement(
                    f"{type(error).__name__}: {error}")
    identity = runtime_identity()
    row.update({
        "case": CASE,
        "steps": n_steps,
        "model_seconds": n_steps * DT_S,
        "schedule": schedule,
        "forward_fd_schedule": forward_schedule,
        **identity,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "precision_policy": repr(get_policy()),
        "measurement_policy_label": measurement_policy_label(
            jax.default_backend(), transcendentals),
        "state_dtype": str(context.initial_state.T.data.dtype),
        "geometry_dtype": str(context.card.recipe.z_coord.dz_ref.dtype),
        "plant": "broken_vjp" if plant else None,
        "production_entry": "LatLonCGridOceanModel._step_jitted",
        "composition": "explicit per-production-step VJP/JVP composition",
        "schedule_scope": "Decision-49 per-step Python composition",
        "per_step_whole_step_rematerialization": True,
        "local_ad_boundary": (
            "separate production primal; unchanged production-step JVP/VJP"),
        "step_transform_compile_footprint": row.get(
            "step_transform_compile_footprint",
            step_transform_footprint(n_steps),
        ),
        "schedule_primal_vs_production": primal_report,
        "forward_primal_vs_production": forward_report,
        "evaluation_primal_vs_production": evaluation_reports,
    })
    return row


def state_bitwise_report(left, right) -> dict:
    """Report exact byte differences between corresponding state leaves."""
    left_path_leaves, left_tree = jax.tree_util.tree_flatten_with_path(left)
    right_path_leaves, right_tree = jax.tree_util.tree_flatten_with_path(right)
    if left_tree != right_tree:
        return {
            "bitwise_equal": False,
            "mismatch_count": None,
            "mismatches": [{"path": "<pytree>", "reason": "treedef differs"}],
        }
    mismatches = []
    for ((left_path, left_leaf), (right_path, right_leaf)) in zip(
            left_path_leaves, right_path_leaves, strict=True):
        require(left_path == right_path, "paired state leaf paths differ")
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        same_layout = (
            left_array.shape == right_array.shape
            and left_array.dtype == right_array.dtype
        )
        same_bytes = same_layout and left_array.tobytes() == right_array.tobytes()
        if same_bytes:
            continue
        entry = {
            "path": jax.tree_util.keystr(left_path),
            "shape": list(left_array.shape),
            "dtype": str(left_array.dtype),
            "max_abs": None,
        }
        if (same_layout and np.issubdtype(left_array.dtype, np.number)
                and np.all(np.isfinite(left_array))
                and np.all(np.isfinite(right_array))):
            entry["max_abs"] = float(np.max(np.abs(
                left_array.astype(np.float64)
                - right_array.astype(np.float64))))
        mismatches.append(entry)
    return {
        "bitwise_equal": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def states_bitwise_equal(left, right) -> bool:
    """Compare all state leaves byte for byte."""
    return bool(state_bitwise_report(left, right)["bitwise_equal"])


def compare_scheduled_primal(left, right) -> dict:
    """Apply the checkpoint scheduler's frozen floating-point value contract."""
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return {
            "within_contract": False,
            "bitwise": False,
            "max_abs": None,
            "rtol": SCHEDULE_RTOL,
            "atol": SCHEDULE_ATOL,
        }
    bitwise = True
    within_contract = True
    max_abs = 0.0
    for left_leaf, right_leaf in zip(
            left_leaves, right_leaves, strict=True):
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        leaf_bitwise = (
            left_array.shape == right_array.shape
            and left_array.dtype == right_array.dtype
            and left_array.tobytes() == right_array.tobytes()
        )
        bitwise = bitwise and leaf_bitwise
        if leaf_bitwise:
            continue
        if left_array.shape != right_array.shape:
            within_contract = False
            continue
        if not (
            np.issubdtype(left_array.dtype, np.inexact)
            and np.issubdtype(right_array.dtype, np.inexact)
        ):
            within_contract = False
            continue
        finite = np.isfinite(left_array) & np.isfinite(right_array)
        if not bool(np.all(finite)):
            within_contract = False
            continue
        max_abs = max(
            max_abs,
            float(np.max(np.abs(left_array - right_array), initial=0.0)),
        )
        within_contract = within_contract and bool(np.allclose(
            left_array,
            right_array,
            rtol=SCHEDULE_RTOL,
            atol=SCHEDULE_ATOL,
        ))
    return {
        "within_contract": within_contract,
        "bitwise": bitwise,
        "max_abs": max_abs,
        "rtol": SCHEDULE_RTOL,
        "atol": SCHEDULE_ATOL,
    }


def current_rss_mib() -> float:
    """Read this process's current resident set from procfs."""
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    raise RuntimeError("/proc/self/status has no VmRSS row")


def gpu1_memory_mib() -> float:
    """Read physical GPU 1 total used memory from nvidia-smi."""
    output = subprocess.check_output(
        [
            "nvidia-smi", "-i", "1", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )
    return float(output.strip().splitlines()[0])


class PeakMemorySampler:
    """Sample host RSS or physical GPU 1 while a blocked reverse pass runs."""

    def __init__(self, backend: str, interval_s: float = 0.05):
        self._read = current_rss_mib if backend == "cpu" else gpu1_memory_mib
        self._interval_s = interval_s
        self._stop = threading.Event()
        self.baseline_mib = self._read()
        self.peak_mib = self.baseline_mib
        self.samples = 1
        self._error = None
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        try:
            while not self._stop.is_set():
                self.peak_mib = max(self.peak_mib, self._read())
                self.samples += 1
                self._stop.wait(self._interval_s)
        except BaseException as error:
            self._error = error
            self._stop.set()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("peak-memory sampler failed") from self._error
        self.peak_mib = max(self.peak_mib, self._read())
        require(self.samples >= 2, "peak-memory sampler took no in-pass sample")


def device_memory_stats(device) -> dict[str, int]:
    """Read the JAX allocator counters required by the GPU memory gate."""
    stats = device.memory_stats()
    require(stats is not None, "JAX device does not expose memory_stats()")
    required = ("bytes_in_use", "peak_bytes_in_use")
    missing = [name for name in required if name not in stats]
    require(not missing, f"JAX device memory_stats missing {missing}")
    result = {
        name: int(value) for name, value in stats.items()
        if isinstance(value, (int, np.integer))
    }
    for name in required:
        require(result[name] >= 0, f"JAX device memory stat {name} is unavailable")
    return result


def require_gpu_allocator_instrument() -> None:
    """Refuse GPU memory rows whose allocator hides live allocations."""
    preallocate = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower()
    allocator = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "").lower()
    require(
        preallocate == "false",
        "GPU resources require XLA_PYTHON_CLIENT_PREALLOCATE=false before JAX import",
    )
    require(
        allocator != "platform",
        "GPU resources refuse XLA_PYTHON_CLIENT_ALLOCATOR=platform because "
        "JAX device memory_stats() is unavailable",
    )


def measure_blocked_reverse(reverse, zero_alpha, backend: str):
    """Time one blocked reverse execution with a single-use sampler."""
    sampler = PeakMemorySampler(backend)
    sampler.start()
    started = time.perf_counter()
    try:
        value, gradient = reverse(zero_alpha)
        jax.block_until_ready((value, gradient))
        wall_seconds = time.perf_counter() - started
    finally:
        sampler.stop()
    return ReversePassMeasurement(value, gradient, wall_seconds, sampler)


def measure_cold_and_warm_reverse(
        reverse, zero_alpha, backend: str, *, after_cold=None):
    """Use distinct samplers and observe the cold pass before the warm pass."""
    cold = measure_blocked_reverse(reverse, zero_alpha, backend)
    cold_observation = after_cold() if after_cold is not None else None
    warm = measure_blocked_reverse(reverse, zero_alpha, backend)
    return cold, warm, cold_observation


def build_compiled_reverse(line):
    """Build the compiled reverse executable used by resource measurements."""
    return jax.jit(jax.value_and_grad(line))


def extrapolate_steps(
        schedule: str, baseline_mib: float, peak6_mib: float,
        n_steps: int) -> float:
    """Apply the preregistered six-step schedule-complexity extrapolation."""
    require(n_steps >= 1, "projection window must contain at least one step")
    require(peak6_mib >= baseline_mib,
            "six-step peak must be at least the sampled baseline")
    incremental = peak6_mib - baseline_mib
    if schedule == "none":
        return baseline_mib + (n_steps / 6.0) * incremental
    if schedule == "uniform":
        return peak6_mib
    if schedule == "binomial":
        return (
            baseline_mib
            + incremental * sqrt_checkpoints(n_steps) / sqrt_checkpoints(6)
        )
    raise ValueError(f"unknown schedule {schedule!r}")


def extrapolate_360(schedule: str, baseline_mib: float, peak6_mib: float) -> float:
    """Retain the round-1 registered 360-step projection entry point."""
    return extrapolate_steps(schedule, baseline_mib, peak6_mib, 360)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_agreement_reference(
        path: Path, *, expect_commit: str, n_steps: int) -> tuple[dict, str]:
    """Load one immutable per-step row used to judge scheduled gradients."""
    digest = _sha256(path)
    row = json.loads(path.read_text(encoding="utf-8"))
    require(row.get("case") == CASE, "agreement reference has wrong card")
    require(row.get("steps") == n_steps, "agreement reference has wrong steps")
    require(row.get("schedule") == "per_step",
            "agreement reference is not the explicit per-step composition")
    require(row.get("status") in ("CONFIRMED", "REFUTED"),
            "agreement reference is not a measured classification")
    require(row.get("step_vjp_boundary_outputs_bitwise") is True,
            "agreement reference failed the VJP boundary byte gate")
    require(row.get("step_jvp_boundary_outputs_bitwise") is True,
            "agreement reference failed the JVP boundary byte gate")
    require(row.get("provenance") == {
        "commit": expect_commit,
        "dirty": False,
        "verified_after_measurement": True,
    }, "agreement reference provenance is not binding")
    for name in ("forward", "finite_difference"):
        value = row.get(name)
        require(isinstance(value, (int, float)) and math.isfinite(value)
                and value != 0.0,
                f"agreement reference {name} is not finite and nonzero")
    return row, digest


def bind_resource_classification(row: dict, gradient_agreement: dict) -> dict:
    """Bind a resource row to both cost validity and unchanged AD bars."""
    row["resource_status"] = classify_resource_row(row)
    row["gradient_agreement_status"] = gradient_agreement["status"]
    row["status"] = (
        row["resource_status"]
        if row["resource_status"] != "CONFIRMED"
        else gradient_agreement["status"]
    )
    return row


def measure_resources(n_steps: int, schedule: str, *,
                      agreement_reference: dict,
                      agreement_reference_path: Path,
                      agreement_reference_sha256: str,
                      transcendentals: str = "libm"):
    """Measure a compiled reverse and its peak footprint in a fresh process."""
    context = build_context(transcendentals=transcendentals)
    zero = jnp.zeros_like(context.initial_state.T.data)
    truth_state = production_rollout(context, zero, n_steps)
    jax.block_until_ready(truth_state)
    reference = build_reference(context, truth_state)
    scheduled_truth = scheduled_rollout(context, zero, n_steps, schedule)
    jax.block_until_ready(scheduled_truth)
    primal_comparison = compare_scheduled_primal(truth_state, scheduled_truth)
    require(
        primal_comparison["within_contract"],
        f"{schedule} schedule primal differs from per-step production beyond "
        f"the checkpoint-schedule contract: {primal_comparison}",
    )
    loss = make_loss(context, reference, n_steps, schedule=schedule)
    base, direction = perturbations(context)

    def line(alpha):
        return loss(base + alpha * direction)

    reverse = build_compiled_reverse(line)
    zero_alpha = jnp.asarray(0.0, dtype=base.dtype)

    backend = jax.default_backend()
    require(backend in ("cpu", "gpu"), f"unsupported backend {backend!r}")
    if backend == "gpu":
        require_gpu_allocator_instrument()
    identity = runtime_identity()
    require(agreement_reference.get("backend") == backend,
            "agreement reference backend differs from resource row")
    require(
        agreement_reference.get("measurement_policy_label")
        == measurement_policy_label(backend, transcendentals),
        "agreement reference precision policy differs from resource row",
    )
    device_stats_before = (
        device_memory_stats(jax.devices()[0]) if backend == "gpu" else None)
    after_cold = (
        (lambda: device_memory_stats(jax.devices()[0]))
        if backend == "gpu" else None)
    cold, warm, device_stats_after = measure_cold_and_warm_reverse(
        reverse, zero_alpha, backend, after_cold=after_cold)
    cold_wall_s = cold.wall_seconds
    value, gradient = warm.value, warm.gradient
    wall_s = warm.wall_seconds
    sampler = cold.sampler if backend == "gpu" else warm.sampler
    gradient_array = np.asarray(gradient)
    if backend == "gpu":
        assert device_stats_before is not None and device_stats_after is not None
        memory_baseline_mib = device_stats_before["bytes_in_use"] / 2**20
        memory_peak_mib = device_stats_after["peak_bytes_in_use"] / 2**20
        memory_samples = 2
    else:
        memory_baseline_mib = sampler.baseline_mib
        memory_peak_mib = sampler.peak_mib
        memory_samples = sampler.samples
    loss_value = float(np.asarray(value))
    gradient_finite = bool(np.all(np.isfinite(gradient_array)))
    gradient_nonzero = bool(np.any(gradient_array != 0.0))
    scheduled_reverse = float(gradient_array)
    gradient_agreement = classify_agreement(
        scheduled_reverse,
        float(agreement_reference["forward"]),
        float(agreement_reference["finite_difference"]),
    )
    row = {
        "resource_schema": RESOURCE_SCHEMA,
        "case": CASE,
        "steps": n_steps,
        "model_seconds": n_steps * DT_S,
        "schedule": schedule,
        **identity,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "precision_policy": repr(get_policy()),
        "measurement_policy_label": measurement_policy_label(
            backend, transcendentals),
        "state_dtype": str(context.initial_state.T.data.dtype),
        "geometry_dtype": str(context.card.recipe.z_coord.dz_ref.dtype),
        "production_entry": "LatLonCGridOceanModel._step_jitted",
        "schedule_scope": SCHEDULE_SCOPE,
        "cold_first_step_segmented": n_steps >= 1,
        "checkpointed_warm_steps": max(0, n_steps - 1),
        "checkpoint_schedule_first_model_step": 2 if n_steps > 1 else None,
        "schedule_primal_bitwise_production": primal_comparison["bitwise"],
        "schedule_primal_comparison": primal_comparison,
        "local_ad_boundary": (
            "separate production primal; unchanged production-step JVP/VJP"),
        "cold_compile_and_reverse_wall_seconds": cold_wall_s,
        "warm_reverse_wall_seconds": wall_s,
        "warm_reverse_timer_span": WARM_TIMER_SPAN,
        "reverse_transform": REVERSE_TRANSFORM,
        "loss": loss_value,
        "gradient_finite": gradient_finite,
        "gradient_nonzero": gradient_nonzero,
        "reverse": gradient_agreement["reverse"],
        "forward": gradient_agreement["forward"],
        "finite_difference": gradient_agreement["finite_difference"],
        "rev_fwd_relative_error": gradient_agreement[
            "rev_fwd_relative_error"],
        "rev_fd_relative_error": gradient_agreement["rev_fd_relative_error"],
        "rev_fwd_bar": REVFWD_TOL,
        "rev_fd_bar": FD_TOL,
        "gradient_agreement_status": gradient_agreement["status"],
        "gradient_dot_abs_sum": agreement_reference.get(
            "gradient_dot_abs_sum"),
        "gradient_direction_norm_product": agreement_reference.get(
            "gradient_direction_norm_product"),
        "rev_fwd_absolute_discrepancy": gradient_agreement[
            "rev_fwd_absolute_discrepancy"],
        "rev_fwd_scale_aware_error": (
            gradient_agreement["rev_fwd_absolute_discrepancy"]
            / max(float(agreement_reference.get(
                "gradient_dot_abs_sum") or 0.0), 1.0e-300)
        ),
        "agreement_reference": str(agreement_reference_path),
        "agreement_reference_sha256": agreement_reference_sha256,
        "memory_kind": (
            "host_rss" if backend == "cpu"
            else "jax_allocator_peak_with_gpu1_nvidia_smi_control"),
        "memory_pass": (
            "warm_reverse" if backend == "cpu"
            else "fresh_process_cold_compile_and_reverse"),
        "memory_baseline_mib": memory_baseline_mib,
        "memory_peak_mib": memory_peak_mib,
        "memory_increment_mib": memory_peak_mib - memory_baseline_mib,
        "memory_samples": memory_samples,
    }
    bind_resource_classification(row, gradient_agreement)
    if backend == "gpu":
        assert device_stats_before is not None and device_stats_after is not None
        peak_growth_mib = (
            device_stats_after["peak_bytes_in_use"]
            - device_stats_before["peak_bytes_in_use"]
        ) / 2**20
        nvidia_increment_mib = sampler.peak_mib - sampler.baseline_mib
        row.update({
            "jax_device_memory_stats_before": device_stats_before,
            "jax_device_memory_stats_after": device_stats_after,
            "jax_device_memory_increment_mib": (
                device_stats_after["peak_bytes_in_use"]
                - device_stats_before["bytes_in_use"]
            ) / 2**20,
            "jax_device_peak_growth_mib": peak_growth_mib,
            "jax_device_peak_scope": "fresh-process lifetime",
            "nvidia_smi_memory_baseline_mib": sampler.baseline_mib,
            "nvidia_smi_memory_peak_mib": sampler.peak_mib,
            "nvidia_smi_memory_increment_mib": nvidia_increment_mib,
            "nvidia_smi_memory_samples": sampler.samples,
            "xla_python_client_preallocate": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE"),
            "xla_python_client_allocator": os.environ.get(
                "XLA_PYTHON_CLIENT_ALLOCATOR"),
        })
        if peak_growth_mib <= 0.0 or nvidia_increment_mib <= 0.0:
            row["resource_status"] = "VACUOUS"
            row["status"] = "VACUOUS"
            row["resource_vacuity_reason"] = (
                "JAX peak and nvidia-smi must both advance during reverse")
    return row


def classify_resource_row(row: dict) -> str:
    """Classify one row, retracting contaminated historical timer schemas."""
    if (
        row.get("resource_schema") != RESOURCE_SCHEMA
        or row.get("reverse_transform") != REVERSE_TRANSFORM
        or row.get("warm_reverse_timer_span") != WARM_TIMER_SPAN
    ):
        return "RETRACTED"
    numeric = (
        row.get("loss"), row.get("cold_compile_and_reverse_wall_seconds"),
        row.get("warm_reverse_wall_seconds"), row.get("memory_baseline_mib"),
        row.get("memory_peak_mib"), row.get("memory_increment_mib"),
    )
    if not all(isinstance(value, (int, float)) for value in numeric):
        return "VACUOUS"
    sampler_visible = bool(
        math.isfinite(row["memory_baseline_mib"])
        and math.isfinite(row["memory_peak_mib"])
        and row["memory_peak_mib"] > row["memory_baseline_mib"]
        and row.get("memory_samples", 0) >= 2
    )
    if not sampler_visible:
        return "VACUOUS"
    passed = bool(
        row.get("gradient_finite")
        and row.get("gradient_nonzero")
        and row.get("schedule_primal_bitwise_production")
        and all(math.isfinite(value) for value in numeric)
        and row["cold_compile_and_reverse_wall_seconds"] > 0.0
        and row["warm_reverse_wall_seconds"] > 0.0
    )
    return "CONFIRMED" if passed else "REFUTED"


def resource_row_passes(row: dict) -> bool:
    """Return the fail-closed resource-row verdict used by the CLI."""
    return (
        row.get("status") == "CONFIRMED"
        and classify_resource_row(row) == "CONFIRMED"
    )


def classify_time_scaling(rows: list[dict], *, plant: bool = False) -> dict:
    """Gate measured warm-time growth; the plant erases all time scaling."""
    if len(rows) < 2:
        return {"status": "VACUOUS", "reason": "at least two rows required"}
    identity = ("case", "schedule", "backend")
    for name in identity:
        if len({row.get(name) for row in rows}) != 1:
            return {"status": "VACUOUS", "reason": f"rows disagree on {name}"}
    if any(classify_resource_row(row) != "CONFIRMED" for row in rows):
        return {
            "status": "VACUOUS",
            "reason": "one or more rows lacks a valid compiled timing span",
        }
    ordered = sorted(rows, key=lambda row: row["steps"])
    steps = np.asarray([row["steps"] for row in ordered], dtype=np.float64)
    times = np.asarray(
        [row["warm_reverse_wall_seconds"] for row in ordered],
        dtype=np.float64,
    )
    evaluated = np.full_like(times, times[0]) if plant else times
    slope, intercept = np.polyfit(steps, evaluated, 1)
    predicted = intercept + slope * steps
    residual = float(np.sum((evaluated - predicted) ** 2))
    total = float(np.sum((evaluated - np.mean(evaluated)) ** 2))
    r_squared = (
        1.0 if total == 0.0 and residual == 0.0
        else 0.0 if total == 0.0
        else 1.0 - residual / total
    )
    passed = bool(slope > 0.0 and evaluated[-1] > evaluated[0])
    return {
        "status": "CONFIRMED" if passed else "REFUTED",
        "reason": (
            "warm reverse time grows with modeled steps"
            if passed else "warm reverse time does not grow with modeled steps"
        ),
        "schedule": ordered[0]["schedule"],
        "backend": ordered[0]["backend"],
        "steps": steps.tolist(),
        "measured_warm_seconds": times.tolist(),
        "evaluated_warm_seconds": evaluated.tolist(),
        "slope_seconds_per_step": float(slope),
        "intercept_seconds": float(intercept),
        "r_squared": float(r_squared),
        "plant": "constant_warm_time" if plant else None,
    }


def classify_gpu_resource_suite(rows: list[dict], *, plant: bool = False) -> dict:
    """Gate only the preregistered 360-step schedule comparisons."""
    by_schedule = {row.get("schedule"): row for row in rows}
    if set(by_schedule) != set(SCHEDULES) or len(rows) != len(SCHEDULES):
        return {"status": "VACUOUS", "reason": "three schedules required"}
    if any(row.get("backend") != "gpu" for row in rows):
        return {"status": "VACUOUS", "reason": "GPU rows required"}
    if {row.get("steps") for row in rows} != {360}:
        return {"status": "VACUOUS", "reason": "360-step rows required"}
    if any(classify_resource_row(row) != "CONFIRMED" for row in rows):
        return {"status": "VACUOUS", "reason": "unconfirmed input row"}
    keys = {
        "jax_allocator": "jax_device_memory_increment_mib",
        "nvidia_smi": "nvidia_smi_memory_increment_mib",
    }
    increments = {}
    for sampler, key in keys.items():
        values = {name: by_schedule[name].get(key) for name in SCHEDULES}
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
            for value in values.values()
        ):
            return {
                "status": "VACUOUS",
                "reason": f"{sampler} cannot resolve positive allocations",
            }
        increments[sampler] = values
    evaluated = {key: dict(value) for key, value in increments.items()}
    if plant:
        for values in evaluated.values():
            values["binomial"] = values["none"]
    ordering = {
        sampler: {
            "binomial_below_none": values["none"] > values["binomial"],
            "uniform_below_none": values["none"] > values["uniform"],
        }
        for sampler, values in evaluated.items()
    }
    return {
        "status": "CONFIRMED" if all(
            all(comparisons.values()) for comparisons in ordering.values()
        ) else "REFUTED",
        "ordering": ordering,
        "increments_mib": increments,
        "evaluated_increments_mib": evaluated,
        "plant": "binomial_label_uses_none_measurement" if plant else None,
    }


def classify_memory_scaling(rows: list[dict], *, plant: bool = False) -> dict:
    """Gate the preregistered across-window memory claims on both samplers."""
    expected = {(schedule, steps) for schedule in SCHEDULES
                for steps in (6, 60, 360)}
    indexed = {(row.get("schedule"), row.get("steps")): row for row in rows}
    if len(rows) != 9 or set(indexed) != expected:
        return {"status": "VACUOUS", "reason": "nine 6/60/360 rows required"}
    if any(row.get("backend") != "gpu" for row in rows):
        return {"status": "VACUOUS", "reason": "GPU rows required"}
    if any(classify_resource_row(row) != "CONFIRMED" for row in rows):
        return {"status": "VACUOUS", "reason": "invalid resource input row"}
    keys = {
        "jax_allocator": "jax_device_memory_increment_mib",
        "nvidia_smi": "nvidia_smi_memory_increment_mib",
    }
    measured = {
        sampler: {
            schedule: {
                steps: indexed[(schedule, steps)].get(key)
                for steps in (6, 60, 360)
            }
            for schedule in SCHEDULES
        }
        for sampler, key in keys.items()
    }
    values = [
        value
        for schedules in measured.values()
        for windows in schedules.values()
        for value in windows.values()
    ]
    if any(
        not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
        for value in values
    ):
        return {"status": "VACUOUS", "reason": "unresolved memory increment"}
    evaluated = {
        sampler: {
            schedule: dict(windows)
            for schedule, windows in schedules.items()
        }
        for sampler, schedules in measured.items()
    }
    if plant:
        for schedules in evaluated.values():
            schedules["none"][360] = schedules["none"][6]
    predicates = {}
    for sampler, schedules in evaluated.items():
        predicates[sampler] = {
            "none_grows_6_60_360": (
                schedules["none"][6] < schedules["none"][60]
                < schedules["none"][360]
            ),
            "uniform_below_none_at_360": (
                schedules["uniform"][360] < schedules["none"][360]
            ),
            "binomial_below_none_at_360": (
                schedules["binomial"][360] < schedules["none"][360]
            ),
        }
    return {
        "status": "CONFIRMED" if all(
            all(checks.values()) for checks in predicates.values()
        ) else "REFUTED",
        "predicates": predicates,
        "increments_mib": measured,
        "evaluated_increments_mib": evaluated,
        "plant": "flatten_none_360_to_6" if plant else None,
    }


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=_repo_root(), text=True).strip()


def require_measurement_stamp(expect_commit: str) -> dict:
    """Refuse stale or dirty measurements before compiling the card."""
    actual = _git_output("rev-parse", "HEAD")
    require(actual == expect_commit,
            f"measurement commit {actual} != expected {expect_commit}")
    status = _git_output("status", "--porcelain")
    require(not status, f"tracked/untracked worktree is not clean:\n{status}")
    return {"commit": actual, "dirty": False}


def require_unchanged_measurement_stamp(opening: dict) -> dict:
    """Recheck provenance after a long measurement, before writing evidence."""
    closing = require_measurement_stamp(opening["commit"])
    require(closing == opening, "measurement provenance changed during the row")
    return {**closing, "verified_after_measurement": True}


def write_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(row, indent=2, sort_keys=True) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=(
            "agreement", "resources", "time-scaling-suite",
            "gpu-resource-suite", "memory-scaling-suite",
        ))
    parser.add_argument("--steps", type=int, choices=AGREEMENT_STEPS)
    parser.add_argument(
        "--schedule", choices=AGREEMENT_COMPOSITIONS)
    parser.add_argument(
        "--plant", choices=(
            "broken_vjp", "constant_time", "gpu_ordering",
            "flat_none_memory",
        ))
    parser.add_argument("--input-row", type=Path, action="append", default=[])
    parser.add_argument(
        "--agreement-row", type=Path,
        help="immutable matching per-step agreement row required by resources",
    )
    parser.add_argument(
        "--transcendentals", choices=("libm", "native"), default="libm",
        help="libm is the certified CPU policy; GPU rows must select native",
    )
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode in ("agreement", "resources") and args.steps is None:
        parser.error(f"{args.mode} mode requires --steps")
    if args.mode in ("agreement", "resources") and args.schedule is None:
        parser.error(f"{args.mode} mode requires --schedule")
    if args.mode == "agreement" and args.plant not in (None, "broken_vjp"):
        parser.error("agreement accepts only the broken_vjp plant")
    if args.mode == "resources" and args.plant is not None:
        parser.error("resources mode does not accept --plant")
    if args.mode == "resources" and args.steps not in RESOURCE_STEPS:
        parser.error(
            f"resources steps must be one of {RESOURCE_STEPS}; got {args.steps}")
    if args.mode == "resources" and args.schedule == "per_step":
        parser.error(
            "resources mode requires one of the registered checkpoint schedules")
    if args.mode == "resources" and args.agreement_row is None:
        parser.error("resources mode requires --agreement-row")
    if args.mode != "resources" and args.agreement_row is not None:
        parser.error("--agreement-row is valid only in resources mode")
    if args.mode == "time-scaling-suite":
        if len(args.input_row) < 2:
            parser.error("time-scaling-suite requires at least two --input-row")
        if args.plant not in (None, "constant_time"):
            parser.error("time-scaling-suite accepts only constant_time plant")
    if args.mode == "gpu-resource-suite":
        if len(args.input_row) != 3:
            parser.error("gpu-resource-suite requires exactly three --input-row")
        if args.plant not in (None, "gpu_ordering"):
            parser.error("gpu-resource-suite accepts only gpu_ordering plant")
    if args.mode == "memory-scaling-suite":
        if len(args.input_row) != 9:
            parser.error(
                "memory-scaling-suite requires exactly nine --input-row")
        if args.plant not in (None, "flat_none_memory"):
            parser.error(
                "memory-scaling-suite accepts only flat_none_memory plant")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    require(not args.output.exists(),
            f"refusing to overwrite evidence: {args.output}")
    provenance = require_measurement_stamp(args.expect_commit)
    if args.mode == "agreement":
        row = measure_agreement(
            args.steps, args.schedule, plant=args.plant == "broken_vjp",
            transcendentals=args.transcendentals)
    elif args.mode == "resources":
        agreement_reference, agreement_sha256 = load_agreement_reference(
            args.agreement_row,
            expect_commit=args.expect_commit,
            n_steps=args.steps,
        )
        row = measure_resources(
            args.steps, args.schedule,
            agreement_reference=agreement_reference,
            agreement_reference_path=args.agreement_row,
            agreement_reference_sha256=agreement_sha256,
            transcendentals=args.transcendentals)
        require(
            _sha256(args.agreement_row) == agreement_sha256,
            "agreement reference changed during resource measurement",
        )
    else:
        input_rows = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in args.input_row
        ]
        if args.mode == "time-scaling-suite":
            row = classify_time_scaling(
                input_rows, plant=args.plant == "constant_time")
        elif args.mode == "gpu-resource-suite":
            row = classify_gpu_resource_suite(
                input_rows, plant=args.plant == "gpu_ordering")
        else:
            row = classify_memory_scaling(
                input_rows, plant=args.plant == "flat_none_memory")
        row["input_rows"] = [str(path) for path in args.input_row]
    row["provenance"] = require_unchanged_measurement_stamp(provenance)
    row["argv"] = list(sys.argv if argv is None else argv)
    write_result(args.output, row)
    print(json.dumps(row, sort_keys=True), flush=True)
    if args.mode == "agreement":
        return 0 if row["status"] == "CONFIRMED" else 1
    if args.mode == "resources":
        return 0 if row["status"] == "CONFIRMED" else 1
    return 0 if row["status"] == "CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
