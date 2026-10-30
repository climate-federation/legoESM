#!/usr/bin/env python3
"""Reverse/forward/FD and adjoint-resource gate for certified ocean cards.

The GYRE protocol is frozen in its round-1 preregistration; the SWG extension
is frozen in ``nemo_testcases_l1_swg_gradgate_round1_prereg.md``.  This module
deliberately contains no physics repair: a poisoned adjoint is an exit-1
measurement.
"""

from __future__ import annotations

import argparse
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
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_nemo_testcase_card,
    build_swg_card,
    validate_nemo_testcase_card_for_execution,
)
from legoesm.ocean.state import OceanSurfaceForcing
from legoesm.ocean.vertical import (
    nemo_qco_live_face_geometry_cgrid,
    nemo_qco_live_t_thickness,
    nemo_qco_resolved_mesh_operands,
)
from legoesm.training.checkpoint_schedule import checkpointed_loop, sqrt_checkpoints

GYRE_CARD = "GYRE-zco"
SWG_CARD = "SWG"
# GYRE retains its registered 60-step time-scaling rows (round 7); SWG its
# registered direct rows.
CARD_STEPS = {GYRE_CARD: (1, 6, 60), SWG_CARD: (1, 24, 720, 1440)}
RESOURCE_STEPS = {
    GYRE_CARD: (1, 6, 60),
    SWG_CARD: (24, 96, 360, 720, 1440),
}
# Legacy GYRE-only names retained for the VJP bisect instrument and tests.
CASE = GYRE_CARD
DT_S = 14400.0
REVFWD_TOL = 1.0e-9
FD_TOL = 0.05
FD_STEP = 1.0e-3
PERTURBATION_K = 1.0e-3
SCHEDULE_RTOL = 1.0e-9
SCHEDULE_ATOL = 1.0e-11
SCHEDULES = ("none", "uniform", "binomial")
AGREEMENT_COMPOSITIONS = ("per_step", *SCHEDULES)
GYRE_FIELDS = ("T", "S", "u", "v")
FIELDS = GYRE_FIELDS
SWG_FIELDS = ("ssh", "u", "v")
GPU_NATIVE_POLICY_LABEL = (
    "GPU, XLA transcendentals — not the certified bit-exact policy"
)
CPU_LIBM_POLICY_LABEL = "CPU, certified fp64/libm bit-exact policy"
SCHEDULE_SCOPE = (
    "component-agnostic checkpoint schedule over production jitted steps"
)
RESOURCE_SCHEMA = "gyre-gradgate-resource-v2-compiled-reverse"
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


class SWGScoredState(NamedTuple):
    ssh: jax.Array
    u: jax.Array
    v: jax.Array


class SWGLossReference(NamedTuple):
    state: SWGScoredState
    ssh_volume: jax.Array
    u_volume: jax.Array
    v_volume: jax.Array
    ssh_variance: jax.Array
    u_variance: jax.Array
    v_variance: jax.Array


class SWGPerturbation(NamedTuple):
    ssh: jax.Array
    u: jax.Array
    v: jax.Array


class GateContext(NamedTuple):
    selector: str
    card: object
    model: LatLonCGridOceanModel
    initial_state: object
    surface_forcings: object
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
        "unsupported gradient measurement policy: "
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


def _load_swg_ladder():
    """Reuse SWG's admitted-record reader and entry reconstruction."""
    path = Path(__file__).with_name("nemo_testcase_l1_swg_ladder.py")
    spec = importlib.util.spec_from_file_location("swg_gradient_entry", path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpointed_production_step(model, card):
    def production_step(state, freshwater, surface):
        return model._step_jitted(state, card.dt_s, freshwater, surface)

    return jax.jit(jax.checkpoint(production_step))


def _build_gyre_context() -> GateContext:
    card = build_nemo_testcase_card(GYRE_CARD)
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
    # Reuse Decision 49 from the GYRE gradient-twin lane: preserve a compiled
    # boundary around every production step and rematerialize that whole step
    # in reverse mode.  This remains a Python loop (never a scan).
    return GateContext(
        GYRE_CARD, card, model, initial, _load_surface_forcings(),
        _checkpointed_production_step(model, card))


def _build_swg_context() -> GateContext:
    """Rebuild the admitted kt=2 stage-1 boundary as a public SWG state."""
    ladder = _load_swg_ladder()
    card = build_swg_card()
    validate_nemo_testcase_card_for_execution(card)
    require(card.dt_s == 3600.0, f"SWG card dt changed: {card.dt_s}")
    cfg = card.recipe.model_config
    require(cfg.outer_integrator == "swe_rk3", "SWG outer dispatch changed")
    require(cfg.vorticity_scheme.scheme == "ens_total", "SWG ENS changed")
    require(cfg.tracer_advection == "none", "SWG acquired tracers")
    require(cfg.physics.vertical_mixing.scheme == "none", "SWG acquired TKE")
    entries = ladder.read_stage_entries(
        ladder.DEFAULT_STAGE_ROOT, record_kts=(2,), stages=(1,))
    entry = entries[2, 1]
    dtype = jnp.float64
    initial = card.recipe.initial_state
    u_native = jnp.asarray(entry["u_k"], dtype=dtype)
    v_native = jnp.asarray(entry["v_k"], dtype=dtype)
    u = jnp.concatenate([jnp.zeros_like(u_native[:, :1]), u_native], axis=1)
    v = jnp.concatenate([jnp.zeros_like(v_native[:1]), v_native], axis=0)
    state = initial._replace(
        eta=initial.eta.replace(data=jnp.asarray(entry["ssh_k"], dtype=dtype)),
        u=initial.u.replace(data=u[..., None]),
        v=initial.v.replace(data=v[..., None]),
        tau_x_prev=jnp.asarray(entry["utau_before"], dtype=dtype),
        tau_y_prev=jnp.asarray(entry["vtau_before"], dtype=dtype),
    )
    forcing = OceanSurfaceForcing(
        tau_i_native=jnp.asarray(entry["utau_now"], dtype=dtype),
        tau_j_native=jnp.asarray(entry["vtau_now"], dtype=dtype),
    )
    rebuilt = ladder._stage_candidate(state, forcing, card)
    require(
        set(rebuilt) == set(entry)
        and all(np.array_equal(rebuilt[name], entry[name]) for name in entry),
        "SWG public state does not rebuild admitted kt=2 stage-1 entry",
    )
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    require(state.eta.data.dtype == dtype, "SWG state is not float64")
    require(card.recipe.z_coord.dz_ref.dtype == dtype,
            "SWG geometry is not float64")

    def surface_forcings(_card, _state, _kt):
        return None, forcing

    return GateContext(
        SWG_CARD, card, model, state, surface_forcings,
        _checkpointed_production_step(model, card))


def build_context(card: str, *, transcendentals: str = "libm") -> GateContext:
    """Dispatch a certified card without a silent fallback."""
    configure_runtime(transcendentals)
    if card == GYRE_CARD:
        return _build_gyre_context()
    if card == SWG_CARD:
        return _build_swg_context()
    raise ValueError(
        f"unknown gradient card {card!r}; expected one of {sorted(CARD_STEPS)}")


def _score_state(state) -> ScoredState:
    return ScoredState(
        state.T.data, state.S.data, state.u.data, state.v.data, state.eta.data)


def _apply_initial_perturbation(context, d_t0):
    """GYRE: perturb the day-0 temperature field only."""
    return context.initial_state._replace(
        T=context.initial_state.T.replace(
            data=context.initial_state.T.data + d_t0))


def _apply_perturbation(context: GateContext, perturbation):
    if context.selector == GYRE_CARD:
        return _apply_initial_perturbation(context, perturbation)
    if context.selector == SWG_CARD:
        return context.initial_state._replace(
            eta=context.initial_state.eta.replace(
                data=context.initial_state.eta.data + perturbation.ssh),
            u=context.initial_state.u.replace(
                data=context.initial_state.u.data + perturbation.u),
            v=context.initial_state.v.replace(
                data=context.initial_state.v.data + perturbation.v),
        )
    raise ValueError(f"unknown gradient card {context.selector!r}")


def zero_perturbation(context: GateContext):
    if context.selector == GYRE_CARD:
        return jnp.zeros_like(context.initial_state.T.data)
    if context.selector == SWG_CARD:
        return SWGPerturbation(
            jnp.zeros_like(context.initial_state.eta.data),
            jnp.zeros_like(context.initial_state.u.data),
            jnp.zeros_like(context.initial_state.v.data),
        )
    raise ValueError(f"unknown gradient card {context.selector!r}")


def _production_step(context, state, kt: int):
    freshwater, surface = context.surface_forcings(context.card, state, kt)
    return context.model._step_jitted(
        state, context.card.dt_s, freshwater, surface)


def production_rollout(context: GateContext, perturbation, n_steps: int):
    """Campaign-order Python loop; every step calls production ``_step_jitted``."""
    state = _apply_perturbation(context, perturbation)
    for kt in range(1, n_steps + 1):
        state = _production_step(context, state, kt)
    return state


def gradient_rollout(context: GateContext, perturbation, n_steps: int):
    """Twin-Decision-49 Python composition of rematerialized production steps."""
    state = _apply_perturbation(context, perturbation)
    for kt in range(1, n_steps + 1):
        freshwater, surface = context.surface_forcings(context.card, state, kt)
        state = context.checkpointed_step(state, freshwater, surface)
    return state


def scheduled_rollout(context: GateContext, perturbation, n_steps: int,
                      schedule: str):
    """Run a shared checkpoint schedule whose body is the production step."""
    if schedule == "per_step":
        return gradient_rollout(context, perturbation, n_steps)
    state = _apply_perturbation(context, perturbation)

    def body(carry):
        kt, step_state = carry
        freshwater, surface = context.surface_forcings(
            context.card, step_state, kt + 1)
        step_state = context.model._step_jitted(
            step_state, context.card.dt_s, freshwater, surface)
        return kt + 1, step_state

    _, state = checkpointed_loop(
        body,
        (jnp.asarray(0, dtype=jnp.int32), state),
        n_steps,
        schedule=schedule,
    )
    return state


def _live_volumes(context: GateContext, truth: ScoredState):
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


def _masked_variance(field, weight):
    mean = jnp.sum(field * weight) / jnp.sum(weight)
    return volume_weighted_mse(field, mean, weight)


def build_reference(context: GateContext, truth_state):
    """Freeze the selected card's truth, wet weights, and normalizers."""
    if context.selector == SWG_CARD:
        truth = SWGScoredState(
            truth_state.eta.data, truth_state.u.data, truth_state.v.data)
        dtype = truth.ssh.dtype
        volumes = {
            "ssh": jnp.asarray(
                truth_state.land_mask.data, dtype=dtype),
            "u": jnp.asarray(
                truth_state.u_mask.data[..., None], dtype=dtype),
            "v": jnp.asarray(
                truth_state.v_mask.data[..., None], dtype=dtype),
        }
        variances = {
            name: _masked_variance(getattr(truth, name), volumes[name])
            for name in SWG_FIELDS
        }
        for name, variance in variances.items():
            value = float(np.asarray(variance))
            require(math.isfinite(value) and value > 0.0,
                    f"{name} truth variance is not positive and finite: {value}")
        return SWGLossReference(
            truth, volumes["ssh"], volumes["u"], volumes["v"],
            variances["ssh"], variances["u"], variances["v"])
    if context.selector != GYRE_CARD:
        raise ValueError(f"unknown gradient card {context.selector!r}")
    truth = _score_state(truth_state)
    t_volume, u_volume, v_volume = _live_volumes(context, truth)
    volumes = {"T": t_volume, "S": t_volume, "u": u_volume, "v": v_volume}
    variances = {}
    for name in GYRE_FIELDS:
        field = getattr(truth, name)
        weight = volumes[name]
        variance = _masked_variance(field, weight)
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


def loss_at_state(reference: LossReference, state):
    """Evaluate the registered GYRE objective from an already-produced after-state."""
    candidate = _score_state(state)
    scores = [
        volume_weighted_mse(
            getattr(candidate, name),
            getattr(reference.state, name),
            getattr(reference, f"{name}_volume"),
        ) / getattr(reference, f"{name}_variance")
        for name in GYRE_FIELDS
    ]
    return jnp.mean(jnp.stack(scores))


def make_loss(context: GateContext, reference, n_steps: int,
              *, schedule: str | None = None):
    """Return the registered normalized after-state loss."""

    def loss(perturbation):
        state = (
            gradient_rollout(context, perturbation, n_steps)
            if schedule is None
            else scheduled_rollout(context, perturbation, n_steps, schedule)
        )
        if context.selector == SWG_CARD:
            candidate = SWGScoredState(
                state.eta.data, state.u.data, state.v.data)
            scores = [
                volume_weighted_mse(
                    getattr(candidate, name), getattr(reference.state, name),
                    getattr(reference, f"{name}_volume"),
                ) / getattr(reference, f"{name}_variance")
                for name in SWG_FIELDS
            ]
            return jnp.mean(jnp.stack(scores))
        if context.selector != GYRE_CARD:
            raise ValueError(f"unknown gradient card {context.selector!r}")
        return loss_at_state(reference, state)

    return loss


def explicit_stepwise_directional_derivatives(
        context, reference: LossReference, base, direction,
        n_steps: int, *, plant: bool = False):
    """Compose one VJP/JVP per production step without a whole-window trace."""
    states = [_apply_initial_perturbation(context, base)]
    for kt in range(1, n_steps + 1):
        states.append(_production_step(context, states[-1], kt))
        jax.block_until_ready(states[-1])

    value, objective_pullback = jax.vjp(
        lambda state: loss_at_state(reference, state), states[-1])
    cotangent = objective_pullback(jnp.ones_like(value))[0]
    replay_reports = {}
    for kt in range(n_steps, 0, -1):
        previous = states[kt - 1]
        replay, step_pullback = jax.vjp(
            lambda state, step=kt: _production_step(context, state, step),
            previous,
        )
        jax.block_until_ready(replay)
        replay_reports[str(kt)] = state_bitwise_report(states[kt], replay)
        cotangent = step_pullback(cotangent)[0]
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
    forward_replay_reports = {}
    for kt in range(1, n_steps + 1):
        replay, tangent = jax.jvp(
            lambda state, step=kt: _production_step(context, state, step),
            (states[kt - 1],),
            (tangent,),
        )
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
        after = _apply_initial_perturbation(context, base + alpha * direction)
        for kt in range(1, n_steps + 1):
            after = _production_step(context, after, kt)
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
        "step_vjp_replay_reports": replay_reports,
        "step_vjp_replay_used_as_primal": False,
        "step_vjp_replays_bitwise": all(
            report["bitwise_equal"] for report in replay_reports.values()),
        "step_jvp_replay_reports": forward_replay_reports,
        "step_jvp_replay_used_as_primal": False,
        "step_jvp_replays_bitwise": all(
            report["bitwise_equal"]
            for report in forward_replay_reports.values()),
    })
    return row


def _normalize_pytree(tree, scale=1.0):
    leaves = jax.tree_util.tree_leaves(tree)
    norm = jnp.sqrt(sum(jnp.vdot(leaf, leaf) for leaf in leaves))
    return jax.tree_util.tree_map(lambda leaf: scale * leaf / norm, tree)


def perturbations(context: GateContext):
    """Return the registered evaluation point and independent unit direction."""
    if context.selector == SWG_CARD:
        keys = jax.random.split(jax.random.PRNGKey(0), 3)
        direction_keys = jax.random.split(jax.random.PRNGKey(42), 3)
        masks = SWGPerturbation(
            jnp.asarray(context.initial_state.land_mask.data, dtype=jnp.float64),
            jnp.asarray(
                context.initial_state.u_mask.data[..., None], dtype=jnp.float64),
            jnp.asarray(
                context.initial_state.v_mask.data[..., None], dtype=jnp.float64),
        )
        shapes = (
            context.initial_state.eta.data.shape,
            context.initial_state.u.data.shape,
            context.initial_state.v.data.shape,
        )
        base = SWGPerturbation(*(
            jax.random.normal(key, shape, dtype=jnp.float64) * mask
            for key, shape, mask in zip(keys, shapes, masks, strict=True)
        ))
        direction = SWGPerturbation(*(
            jax.random.normal(key, shape, dtype=jnp.float64) * mask
            for key, shape, mask in zip(
                direction_keys, shapes, masks, strict=True)
        ))
        return (
            _normalize_pytree(base, PERTURBATION_K),
            _normalize_pytree(direction),
        )
    if context.selector != GYRE_CARD:
        raise ValueError(f"unknown gradient card {context.selector!r}")
    wet = (jnp.asarray(
        context.card.recipe.z_coord.is_active, dtype=jnp.float64)
        * context.initial_state.land_mask.data[..., None])
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


def _tree_axpy(base, alpha, direction):
    return jax.tree_util.tree_map(
        lambda base_leaf, direction_leaf: base_leaf + alpha * direction_leaf,
        base,
        direction,
    )


def _tree_vdot(left, right) -> float:
    return float(sum(
        float(np.asarray(jnp.vdot(a, b)))
        for a, b in zip(jax.tree_util.tree_leaves(left),
                        jax.tree_util.tree_leaves(right), strict=True)))


def _tree_norm(tree) -> float:
    return math.sqrt(sum(
        float(np.asarray(jnp.vdot(leaf, leaf)))
        for leaf in jax.tree_util.tree_leaves(tree)))


def directional_derivatives(
        forward_loss, base, direction, *, reverse_loss=None, plant=False):
    """Measure one direction and its full-gradient cancellation scales.

    ``base``/``direction`` may be one array (GYRE) or a pytree of arrays
    (SWG).  Reverse mode is the full input VJP dotted with ``direction``; the
    direct unit tests pin this against the scalar-line representation.
    ``reverse_loss`` permits a reverse-only checkpoint schedule to be checked
    against the bit-identical forward-mode-capable composition.
    """
    dtype = jax.tree_util.tree_leaves(base)[0].dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    reverse_loss = forward_loss if reverse_loss is None else reverse_loss

    def forward_line(alpha):
        return forward_loss(_tree_axpy(base, alpha, direction))

    def reverse_objective(value):
        reverse_input = (
            jax.tree_util.tree_map(broken_vjp_identity, value) if plant
            else value)
        return reverse_loss(reverse_input)

    gradient = jax.grad(reverse_objective)(base)
    reverse = _tree_vdot(gradient, direction)
    _, forward_value = jax.jvp(
        forward_line, (zero,), (jnp.asarray(1.0, dtype),))
    forward = float(np.asarray(forward_value))
    plus = float(np.asarray(forward_line(jnp.asarray(FD_STEP, dtype))))
    minus = float(np.asarray(forward_line(jnp.asarray(-FD_STEP, dtype))))
    finite_difference = (plus - minus) / (2.0 * FD_STEP)
    row = classify_agreement(reverse, forward, finite_difference)
    dot_abs_sum = float(sum(
        float(np.asarray(jnp.sum(jnp.abs(g * d))))
        for g, d in zip(jax.tree_util.tree_leaves(gradient),
                        jax.tree_util.tree_leaves(direction), strict=True)))
    norm_product = _tree_norm(gradient) * _tree_norm(direction)
    absolute_discrepancy = abs(reverse - forward)
    row.update({
        "gradient_dot_abs_sum": dot_abs_sum,
        "gradient_direction_norm_product": norm_product,
        "rev_fwd_absolute_discrepancy": absolute_discrepancy,
        "rev_fwd_scale_aware_error": (
            absolute_discrepancy / max(dot_abs_sum, 1.0e-300)),
    })
    return row


def _measure_gyre_agreement(context: GateContext, n_steps: int, schedule: str,
                            *, plant: bool) -> tuple[dict, dict]:
    """GYRE: explicit per-production-step VJP/JVP composition (round 7)."""
    zero = zero_perturbation(context)
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
    row = classify_agreement(math.nan, math.nan, math.nan)
    evaluation_reports = {}
    if not primal_report["bitwise_equal"]:
        row["error"] = (
            f"{schedule} segmented schedule primal differs bitwise from "
            "the production Python loop")
    elif not forward_report["bitwise_equal"]:
        row["error"] = (
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
            row["error"] = (
                "scheduled primal differs bitwise from production at "
                + ", ".join(bad_evaluation_reports))
        else:
            try:
                row = explicit_stepwise_directional_derivatives(
                    context, reference, base, direction, n_steps,
                    plant=plant)
            except (TypeError, ValueError, RuntimeError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
    extra = {
        "schedule": schedule,
        "forward_fd_schedule": forward_schedule,
        "composition": "explicit per-production-step VJP/JVP composition",
        "schedule_scope": "Decision-49 per-step Python composition",
        "per_step_whole_step_rematerialization": True,
        "schedule_primal_vs_production": primal_report,
        "forward_primal_vs_production": forward_report,
        "evaluation_primal_vs_production": evaluation_reports,
    }
    return row, extra


def _measure_swg_agreement(context: GateContext, n_steps: int, schedule: str,
                           *, plant: bool) -> tuple[dict, dict]:
    """SWG: directional reverse/forward/FD over the per-step Python loop."""
    zero = zero_perturbation(context)
    truth_state = production_rollout(context, zero, n_steps)
    jax.block_until_ready(truth_state)
    # Preserve the production JIT boundary at every step.  A scan around the
    # already-jitted step changes CPU rounding after one step; that rejected
    # preregistered composition is retained in the SWG round-1 receipt.
    agreement_schedule = None
    gradient_truth = gradient_rollout(context, zero, n_steps)
    jax.block_until_ready(gradient_truth)
    require(states_bitwise_equal(truth_state, gradient_truth),
            "checkpointed per-step primal differs from production")
    reference = build_reference(context, truth_state)
    loss = make_loss(
        context, reference, n_steps, schedule=agreement_schedule)
    base, direction = perturbations(context)
    row = classify_agreement(math.nan, math.nan, math.nan)
    try:
        row = directional_derivatives(loss, base, direction, plant=plant)
    except (TypeError, ValueError, RuntimeError) as error:
        row["error"] = f"{type(error).__name__}: {error}"
    extra = {
        "schedule": schedule,
        "composition": (
            "per-step Python loop" if agreement_schedule is None
            else f"checkpointed_loop:{agreement_schedule}"),
    }
    return row, extra


def measure_agreement(card: str, n_steps: int, schedule: str = "per_step",
                      *, plant=False, transcendentals: str = "libm"):
    """Build and evaluate one certified-card agreement row."""
    require(schedule in AGREEMENT_COMPOSITIONS,
            f"agreement schedule must be one of {AGREEMENT_COMPOSITIONS}")
    context = build_context(card, transcendentals=transcendentals)
    require(n_steps in CARD_STEPS[card],
            f"{card} agreement steps must be one of {CARD_STEPS[card]}")
    if card == GYRE_CARD:
        row, extra = _measure_gyre_agreement(
            context, n_steps, schedule, plant=plant)
    else:
        row, extra = _measure_swg_agreement(
            context, n_steps, schedule, plant=plant)
    identity = runtime_identity()
    row.update({
        "card_selector": card,
        "case": context.card.case,
        "steps": n_steps,
        "model_seconds": n_steps * context.card.dt_s,
        **identity,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "precision_policy": repr(get_policy()),
        "measurement_policy_label": measurement_policy_label(
            identity["backend"], transcendentals),
        "state_dtype": str(context.initial_state.T.data.dtype),
        "geometry_dtype": str(context.card.recipe.z_coord.dz_ref.dtype),
        "plant": "broken_vjp" if plant else None,
        "production_entry": "LatLonCGridOceanModel._step_jitted",
        "initial_boundary": (
            "admitted kt=2 stage-1 entry" if card == SWG_CARD
            else "certified day-0 state"),
        "loss_fields": list(SWG_FIELDS if card == SWG_CARD else GYRE_FIELDS),
        **extra,
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
    result = {name: int(value) for name, value in stats.items()
              if isinstance(value, (int, np.integer))}
    for name in required:
        require(result[name] >= 0, f"JAX device memory stat {name} is unavailable")
    return result


def require_gpu_allocator_instrument() -> None:
    """Refuse GPU memory rows whose allocator hides live adjoint allocations."""
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
    """Time one blocked reverse execution with a single-use memory sampler."""
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
    reverse, zero_alpha, backend: str, *, after_cold=None,
):
    """Use distinct samplers and observe the cold pass before running warm."""
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


def extrapolate_memory(schedule: str, source_steps: int, target_steps: int,
                       baseline_mib: float, peak_mib: float) -> float:
    """Scale measured incremental tape memory by schedule complexity."""
    if source_steps <= 0 or target_steps < source_steps:
        raise ValueError("memory extrapolation requires target >= positive source")
    incremental = max(0.0, peak_mib - baseline_mib)
    if schedule == "none":
        scale = target_steps / source_steps
    elif schedule == "uniform":
        scale = 1.0
    elif schedule == "binomial":
        scale = (math.isqrt(target_steps - 1) + 1) / (
            math.isqrt(source_steps - 1) + 1)
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    return baseline_mib + scale * incremental


def classify_resources(
    loss: float,
    gradient_finite: bool,
    gradient_nonzero: bool,
    *,
    cold_wall_s: float,
    warm_wall_s: float,
    baseline_mib: float,
    peak_mib: float,
    samples: int,
) -> str:
    """Apply every preregistered resource-row acceptance condition."""
    sampler_visible = (
        math.isfinite(baseline_mib)
        and math.isfinite(peak_mib)
        and peak_mib > baseline_mib
        and samples >= 2
    )
    if not sampler_visible:
        return "VACUOUS"
    passed = (
        math.isfinite(loss)
        and gradient_finite
        and gradient_nonzero
        and math.isfinite(cold_wall_s)
        and cold_wall_s > 0.0
        and math.isfinite(warm_wall_s)
        and warm_wall_s > 0.0
        and sampler_visible
    )
    return "CONFIRMED" if passed else "REFUTED"


def classify_resource_row(row: dict) -> str:
    """Classify one written row, retracting contaminated historical timer schemas."""
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
    if not math.isfinite(row["memory_increment_mib"]):
        return "REFUTED"
    return classify_resources(
        float(row["loss"]),
        bool(row.get("gradient_finite")),
        bool(row.get("gradient_nonzero")),
        cold_wall_s=float(row["cold_compile_and_reverse_wall_seconds"]),
        warm_wall_s=float(row["warm_reverse_wall_seconds"]),
        baseline_mib=float(row["memory_baseline_mib"]),
        peak_mib=float(row["memory_peak_mib"]),
        samples=int(row.get("memory_samples", 0)),
    )


def resource_row_passes(row: dict) -> bool:
    """Return the fail-closed resource-row verdict used by the CLI."""
    return classify_resource_row(row) == "CONFIRMED"


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


def classify_resource_suite(rows: list[dict], *, plant: bool = False) -> dict:
    """Require both GPU samplers to resolve the registered schedule ordering."""
    by_schedule = {row.get("schedule"): row for row in rows}
    required = set(SCHEDULES)
    if set(by_schedule) != required or len(rows) != len(required):
        return {
            "status": "VACUOUS",
            "reason": "resource suite requires exactly none, uniform, and binomial",
        }

    identity_fields = ("card_selector", "steps", "backend")
    for name in identity_fields:
        if len({row.get(name) for row in rows}) != 1:
            return {
                "status": "VACUOUS",
                "reason": f"resource rows disagree on {name}",
            }
    if rows[0].get("backend") != "gpu":
        return {
            "status": "VACUOUS",
            "reason": "dual-sampler ordering is a GPU-only resource verdict",
        }
    if any(row.get("status") != "CONFIRMED" for row in rows):
        return {
            "status": "VACUOUS",
            "reason": "one or more resource rows is not individually measurable",
        }

    keys = {
        "jax_allocator": "jax_device_memory_increment_mib",
        "nvidia_smi": "nvidia_smi_memory_increment_mib",
    }
    increments: dict[str, dict[str, float]] = {}
    for sampler, key in keys.items():
        values: dict[str, float] = {}
        for schedule, row in by_schedule.items():
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                return {
                    "status": "VACUOUS",
                    "reason": f"{sampler} did not report finite increments",
                }
            values[schedule] = float(value)
        if any(value <= 0.0 for value in values.values()):
            return {
                "status": "VACUOUS",
                "reason": f"{sampler} cannot see positive adjoint allocations",
                "increments_mib": {sampler: values},
            }
        increments[sampler] = values

    peak_growth = {
        schedule: by_schedule[schedule].get("jax_device_peak_growth_mib")
        for schedule in SCHEDULES
    }
    if any(
        not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
        for value in peak_growth.values()
    ):
        return {
            "status": "VACUOUS",
            "reason": "JAX process peak did not advance during every reverse pass",
            "jax_device_peak_growth_mib": peak_growth,
        }

    evaluated = {
        sampler: dict(values) for sampler, values in increments.items()
    }
    if plant:
        for values in evaluated.values():
            values["uniform"] = values["none"]
    ordering = {
        sampler: (
            values["none"] > values["binomial"] > values["uniform"])
        for sampler, values in evaluated.items()
    }
    return {
        "status": "CONFIRMED" if all(ordering.values()) else "REFUTED",
        "reason": (
            "both samplers resolve none > binomial > uniform"
            if all(ordering.values())
            else "registered none > binomial > uniform ordering failed"
        ),
        "increments_mib": increments,
        "evaluated_increments_mib": evaluated,
        "ordering": ordering,
        "plant": "uniform_label_uses_none_measurement" if plant else None,
    }


# The GYRE round-7 name for the same dual-sampler ordering verdict.
classify_gpu_resource_suite = classify_resource_suite


def measure_resources(card: str, n_steps: int, schedule: str, *,
                      transcendentals: str = "libm"):
    """Measure a fresh-process GPU reverse or a blocked warm CPU reverse."""
    if card not in CARD_STEPS:
        raise ValueError(
            f"unknown gradient card {card!r}; expected one of "
            f"{sorted(CARD_STEPS)}")
    require(n_steps in RESOURCE_STEPS[card],
            f"{card} resource steps must be one of {RESOURCE_STEPS[card]}")
    require(schedule in SCHEDULES,
            f"resources require one of the registered schedules {SCHEDULES}")
    context = build_context(card, transcendentals=transcendentals)
    zero = zero_perturbation(context)
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
        return loss(_tree_axpy(base, alpha, direction))

    # The measured object is a compiled reverse executable.  Without this
    # outer JIT, every invocation retraces the transformed Python function and
    # the alleged warm-pass timer includes a step-independent trace/compile.
    reverse = build_compiled_reverse(line)
    dtype = jax.tree_util.tree_leaves(base)[0].dtype
    zero_alpha = jnp.asarray(0.0, dtype=dtype)

    backend = jax.default_backend()
    require(backend in ("cpu", "gpu"), f"unsupported backend {backend!r}")
    if backend == "gpu":
        require_gpu_allocator_instrument()
    identity = runtime_identity()

    # JAX exposes a process-lifetime allocator peak, but no peak-reset API.
    # Each GPU row therefore uses its fresh process as the reset boundary and
    # samples the first (cold) reverse.  CPU retains the round-1 warm-RSS
    # protocol because procfs reports a current value that the sampler can
    # reset locally after compilation.
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
    # GPU's process-lifetime allocator peak belongs to the first call in the
    # fresh process, so pair it with the simultaneous cold nvidia-smi sample.
    # CPU RSS has a resettable current value, so retain the warm-pass sample.
    sampler = cold.sampler if backend == "gpu" else warm.sampler
    gradient_array = np.asarray(gradient)
    loss_value = float(np.asarray(value))
    gradient_finite = bool(np.all(np.isfinite(gradient_array)))
    gradient_nonzero = bool(np.any(gradient_array != 0.0))
    if backend == "gpu":
        assert device_stats_before is not None and device_stats_after is not None
        memory_baseline_mib = device_stats_before["bytes_in_use"] / 2**20
        memory_peak_mib = device_stats_after["peak_bytes_in_use"] / 2**20
        memory_samples = 2
    else:
        memory_baseline_mib = sampler.baseline_mib
        memory_peak_mib = sampler.peak_mib
        memory_samples = sampler.samples

    row = {
        "resource_schema": RESOURCE_SCHEMA,
        "card_selector": card,
        "case": context.card.case,
        "steps": n_steps,
        "model_seconds": n_steps * context.card.dt_s,
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
        "schedule_primal_bitwise_production": primal_comparison["bitwise"],
        "schedule_primal_comparison": primal_comparison,
        "cold_compile_and_reverse_wall_seconds": cold_wall_s,
        "warm_reverse_wall_seconds": wall_s,
        "warm_reverse_timer_span": WARM_TIMER_SPAN,
        "reverse_transform": REVERSE_TRANSFORM,
        "loss": loss_value,
        "gradient_finite": gradient_finite,
        "gradient_nonzero": gradient_nonzero,
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
        "initial_boundary": (
            "admitted kt=2 stage-1 entry" if card == SWG_CARD
            else "certified day-0 state"),
    }
    row["status"] = classify_resource_row(row)
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
            row["status"] = "VACUOUS"
            row["resource_vacuity_reason"] = (
                "JAX peak and nvidia-smi must both advance during the reverse pass")
    return row


def measure_resource_suite(paths: list[Path], *, plant: bool = False) -> dict:
    """Classify three already-written GPU resource rows without rerunning them."""
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    row = classify_resource_suite(rows, plant=plant)
    row.update({
        "card_selector": rows[0].get("card_selector") if rows else None,
        "steps": rows[0].get("steps") if rows else None,
        "backend": rows[0].get("backend") if rows else None,
        "resource_rows": [str(path) for path in paths],
        "input_provenance_commits": sorted({
            commit for input_row in rows
            if (commit := input_row.get("provenance", {}).get("commit"))
            is not None
        }),
    })
    return row


def measure_time_scaling_suite(paths: list[Path], *, plant: bool = False) -> dict:
    """Classify already-written resource rows' warm-time growth with steps."""
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    row = classify_time_scaling(rows, plant=plant)
    row.update({
        "card_selector": rows[0].get("card_selector") if rows else None,
        "resource_rows": [str(path) for path in paths],
        "input_provenance_commits": sorted({
            commit for input_row in rows
            if (commit := input_row.get("provenance", {}).get("commit"))
            is not None
        }),
    })
    return row


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
            "agreement", "resources", "resource-suite", "time-scaling-suite"))
    parser.add_argument("--card", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--schedule", choices=AGREEMENT_COMPOSITIONS)
    parser.add_argument(
        "--plant", choices=(
            "broken_vjp", "uniform_label_uses_none", "constant_time"))
    parser.add_argument("--row", action="append", type=Path, default=[])
    parser.add_argument(
        "--transcendentals", choices=("libm", "native"), default="libm",
        help="libm is the certified CPU policy; GPU rows must select native",
    )
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "time-scaling-suite":
        if args.steps is not None:
            parser.error("time-scaling-suite rows span several --steps")
    elif args.steps is None:
        parser.error(f"{args.mode} mode requires --steps")
    elif args.steps <= 0:
        parser.error("--steps must be positive")
    if args.mode == "resources" and args.schedule is None:
        parser.error("resources mode requires --schedule")
    if args.mode == "resources" and args.schedule == "per_step":
        parser.error(
            "resources mode requires one of the registered checkpoint schedules")
    if args.mode == "agreement" and args.schedule is None:
        args.schedule = "per_step"
    if args.mode not in ("agreement", "resources") and args.schedule is not None:
        parser.error(f"{args.mode} mode does not accept --schedule")
    if args.mode == "resources" and args.plant is not None:
        parser.error("resources mode does not accept --plant")
    if args.mode == "agreement" and args.plant not in (None, "broken_vjp"):
        parser.error("agreement mode accepts only --plant broken_vjp")
    if args.mode == "resource-suite":
        if len(args.row) != 3:
            parser.error("resource-suite mode requires exactly three --row paths")
        if args.plant not in (None, "uniform_label_uses_none"):
            parser.error(
                "resource-suite mode accepts only --plant uniform_label_uses_none")
    elif args.mode == "time-scaling-suite":
        if len(args.row) < 2:
            parser.error("time-scaling-suite requires at least two --row paths")
        if args.plant not in (None, "constant_time"):
            parser.error("time-scaling-suite accepts only --plant constant_time")
    elif args.row:
        parser.error(f"{args.mode} mode does not accept --row")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    require(not args.output.exists(),
            f"refusing to overwrite evidence: {args.output}")
    provenance = require_measurement_stamp(args.expect_commit)
    if args.mode == "agreement":
        row = measure_agreement(
            args.card, args.steps, args.schedule,
            plant=args.plant == "broken_vjp",
            transcendentals=args.transcendentals)
    elif args.mode == "resources":
        row = measure_resources(
            args.card, args.steps, args.schedule,
            transcendentals=args.transcendentals)
    elif args.mode == "resource-suite":
        row = measure_resource_suite(
            args.row, plant=args.plant == "uniform_label_uses_none")
        require(args.card in CARD_STEPS, f"unknown gradient card {args.card!r}")
        require(
            row["card_selector"] == args.card and row["steps"] == args.steps,
            "resource-suite CLI card/steps do not match input rows",
        )
    else:
        row = measure_time_scaling_suite(
            args.row, plant=args.plant == "constant_time")
        require(args.card in CARD_STEPS, f"unknown gradient card {args.card!r}")
        require(
            row["card_selector"] == args.card,
            "time-scaling-suite CLI card does not match input rows",
        )
    row["provenance"] = require_unchanged_measurement_stamp(provenance)
    row["argv"] = list(sys.argv if argv is None else argv)
    write_result(args.output, row)
    print(json.dumps(row, sort_keys=True), flush=True)
    return 0 if row["status"] == "CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
