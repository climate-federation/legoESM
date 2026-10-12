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
from legoesm.training.checkpoint_schedule import checkpointed_loop

GYRE_CARD = "GYRE-zco"
SWG_CARD = "SWG"
CARD_STEPS = {GYRE_CARD: (1, 6), SWG_CARD: (1, 24, 720)}
RESOURCE_STEPS = {GYRE_CARD: (1, 6), SWG_CARD: (24, 720)}
REVFWD_TOL = 1.0e-9
FD_TOL = 0.05
FD_STEP = 1.0e-3
PERTURBATION_K = 1.0e-3
SCHEDULE_RTOL = 1.0e-9
SCHEDULE_ATOL = 1.0e-11
SCHEDULES = ("none", "uniform", "binomial")
GYRE_FIELDS = ("T", "S", "u", "v")
SWG_FIELDS = ("ssh", "u", "v")
GPU_NATIVE_POLICY_LABEL = (
    "GPU, XLA transcendentals — not the certified bit-exact policy"
)
CPU_LIBM_POLICY_LABEL = "CPU, certified fp64/libm bit-exact policy"


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
    require(card.dt_s == 14400.0, f"card dt changed: {card.dt_s}")
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


def _apply_perturbation(context: GateContext, perturbation):
    if context.selector == GYRE_CARD:
        return context.initial_state._replace(
            T=context.initial_state.T.replace(
                data=context.initial_state.T.data + perturbation))
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


def production_rollout(context: GateContext, perturbation, n_steps: int):
    """Campaign-order Python loop; every step calls production ``_step_jitted``."""
    state = _apply_perturbation(context, perturbation)
    for kt in range(1, n_steps + 1):
        freshwater, surface = context.surface_forcings(context.card, state, kt)
        state = context.model._step_jitted(
            state,
            context.card.dt_s,
            freshwater,
            surface,
        )
    return state


def gradient_rollout(context: GateContext, perturbation, n_steps: int):
    """Twin-Decision-49 per-step remat around production ``_step_jitted``."""
    state = _apply_perturbation(context, perturbation)
    for kt in range(1, n_steps + 1):
        freshwater, surface = context.surface_forcings(context.card, state, kt)
        state = context.checkpointed_step(state, freshwater, surface)
    return state


def scheduled_rollout(context: GateContext, perturbation, n_steps: int,
                      schedule: str):
    """Run the shared component-agnostic checkpoint schedule over production steps."""
    state = _apply_perturbation(context, perturbation)

    def body(carry):
        kt, step_state = carry
        freshwater, surface = context.surface_forcings(
            context.card, step_state, kt + 1)
        step_state = context.model._step_jitted(
            step_state,
            context.card.dt_s,
            freshwater,
            surface,
        )
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


def make_loss(context: GateContext, reference, n_steps: int,
              *, schedule: str | None = None):
    """Return the registered normalized after-state loss."""
    rollout = gradient_rollout if schedule is None else scheduled_rollout

    def loss(perturbation):
        if schedule is None:
            state = rollout(context, perturbation, n_steps)
        else:
            state = rollout(context, perturbation, n_steps, schedule)
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
        candidate = _score_state(state)
        scores = []
        for name in GYRE_FIELDS:
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
    nonzero = all(value != 0.0 for value in (reverse, forward, finite_difference))
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
    }


def directional_derivatives(loss, base, direction, *, plant=False):
    """Measure the scalar line ``loss(base + alpha*direction)`` three ways.

    Reverse mode with respect to ``alpha`` is exactly the full input VJP dotted
    with ``direction``, but avoids materializing that 22x32x30 result.  The
    direct unit test pins this equality and prevents this representation from
    silently changing the requested directional derivative.
    """
    dtype = jax.tree_util.tree_leaves(base)[0].dtype
    zero = jnp.asarray(0.0, dtype=dtype)

    def line(alpha):
        value = jax.tree_util.tree_map(
            lambda base_leaf, direction_leaf: (
                base_leaf + alpha * direction_leaf),
            base,
            direction,
        )
        return loss(value)

    reverse_line = (
        (lambda alpha: line(broken_vjp_identity(alpha))) if plant else line)
    reverse = float(np.asarray(jax.grad(reverse_line)(zero)))
    _, forward_value = jax.jvp(line, (zero,), (jnp.asarray(1.0, dtype),))
    forward = float(np.asarray(forward_value))
    plus = float(np.asarray(line(jnp.asarray(FD_STEP, dtype))))
    minus = float(np.asarray(line(jnp.asarray(-FD_STEP, dtype))))
    finite_difference = (plus - minus) / (2.0 * FD_STEP)
    return classify_agreement(reverse, forward, finite_difference)


def measure_agreement(card: str, n_steps: int, *, plant=False,
                      transcendentals: str = "libm"):
    """Build and evaluate one certified-card agreement row."""
    context = build_context(card, transcendentals=transcendentals)
    require(n_steps in CARD_STEPS[card],
            f"{card} agreement steps must be one of {CARD_STEPS[card]}")
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
    row.update({
        "card_selector": card,
        "case": context.card.case,
        "steps": n_steps,
        "model_seconds": n_steps * context.card.dt_s,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "precision_policy": repr(get_policy()),
        "measurement_policy_label": measurement_policy_label(
            jax.default_backend(), transcendentals),
        "state_dtype": str(context.initial_state.T.data.dtype),
        "geometry_dtype": str(context.card.recipe.z_coord.dz_ref.dtype),
        "plant": "broken_vjp" if plant else None,
        "production_entry": "LatLonCGridOceanModel._step_jitted",
        "composition": (
            "per-step Python loop" if agreement_schedule is None
            else f"checkpointed_loop:{agreement_schedule}"),
        "initial_boundary": (
            "admitted kt=2 stage-1 entry" if card == SWG_CARD
            else "certified day-0 state"),
        "loss_fields": list(SWG_FIELDS if card == SWG_CARD else GYRE_FIELDS),
    })
    return row


def states_bitwise_equal(left, right) -> bool:
    """Compare all floating and integer state leaves without summary rounding."""
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return False
    return all(np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(left_leaves, right_leaves, strict=True))


def compare_scheduled_primal(left, right) -> dict:
    """Apply the repository's frozen checkpoint-schedule value contract."""
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
        leaf_bitwise = np.array_equal(left_array, right_array)
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
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, self._read())
            self.samples += 1
            self._stop.wait(self._interval_s)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        self.peak_mib = max(self.peak_mib, self._read())


def extrapolate_360(schedule: str, baseline_mib: float, peak6_mib: float) -> float:
    """Apply the preregistered schedule-complexity extrapolation."""
    incremental = max(0.0, peak6_mib - baseline_mib)
    if schedule == "none":
        return baseline_mib + 60.0 * incremental
    if schedule == "uniform":
        return peak6_mib
    if schedule == "binomial":
        return baseline_mib + incremental * 19.0 / 3.0
    raise ValueError(f"unknown schedule {schedule!r}")


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
    passed = (
        math.isfinite(loss)
        and gradient_finite
        and gradient_nonzero
        and math.isfinite(cold_wall_s)
        and cold_wall_s > 0.0
        and math.isfinite(warm_wall_s)
        and warm_wall_s > 0.0
        and math.isfinite(baseline_mib)
        and math.isfinite(peak_mib)
        and peak_mib >= baseline_mib
        and samples >= 2
    )
    return "CONFIRMED" if passed else "REFUTED"


def measure_resources(card: str, n_steps: int, schedule: str, *,
                      transcendentals: str = "libm"):
    """Warm once, then measure one blocked reverse execution and peak memory."""
    context = build_context(card, transcendentals=transcendentals)
    require(n_steps in RESOURCE_STEPS[card],
            f"{card} resource steps must be one of {RESOURCE_STEPS[card]}")
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
        value = jax.tree_util.tree_map(
            lambda base_leaf, direction_leaf: (
                base_leaf + alpha * direction_leaf),
            base,
            direction,
        )
        return loss(value)

    reverse = jax.value_and_grad(line)
    dtype = jax.tree_util.tree_leaves(base)[0].dtype
    zero_alpha = jnp.asarray(0.0, dtype=dtype)

    cold_started = time.perf_counter()
    cold_value, cold_gradient = reverse(zero_alpha)
    jax.block_until_ready((cold_value, cold_gradient))
    cold_wall_s = time.perf_counter() - cold_started

    backend = jax.default_backend()
    require(backend in ("cpu", "gpu"), f"unsupported backend {backend!r}")
    if backend == "gpu":
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1",
                "GPU measurements require CUDA_VISIBLE_DEVICES=1")
    sampler = PeakMemorySampler(backend)
    sampler.start()
    started = time.perf_counter()
    value, gradient = reverse(zero_alpha)
    jax.block_until_ready((value, gradient))
    wall_s = time.perf_counter() - started
    sampler.stop()
    gradient_array = np.asarray(gradient)
    loss_value = float(np.asarray(value))
    gradient_finite = bool(np.all(np.isfinite(gradient_array)))
    gradient_nonzero = bool(np.any(gradient_array != 0.0))
    row = {
        "card_selector": card,
        "case": context.card.case,
        "steps": n_steps,
        "model_seconds": n_steps * context.card.dt_s,
        "schedule": schedule,
        "backend": backend,
        "device": str(jax.devices()[0]),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "precision_policy": repr(get_policy()),
        "measurement_policy_label": measurement_policy_label(
            backend, transcendentals),
        "state_dtype": str(context.initial_state.T.data.dtype),
        "geometry_dtype": str(context.card.recipe.z_coord.dz_ref.dtype),
        "production_entry": "LatLonCGridOceanModel._step_jitted",
        "schedule_primal_bitwise_production": primal_comparison["bitwise"],
        "schedule_primal_comparison": primal_comparison,
        "cold_compile_and_reverse_wall_seconds": cold_wall_s,
        "warm_reverse_wall_seconds": wall_s,
        "loss": loss_value,
        "gradient_finite": gradient_finite,
        "gradient_nonzero": gradient_nonzero,
        "memory_kind": "host_rss" if backend == "cpu" else "gpu1_nvidia_smi",
        "memory_baseline_mib": sampler.baseline_mib,
        "memory_peak_mib": sampler.peak_mib,
        "memory_increment_mib": sampler.peak_mib - sampler.baseline_mib,
        "memory_samples": sampler.samples,
        "status": classify_resources(
            loss_value,
            gradient_finite,
            gradient_nonzero,
            cold_wall_s=cold_wall_s,
            warm_wall_s=wall_s,
            baseline_mib=sampler.baseline_mib,
            peak_mib=sampler.peak_mib,
            samples=sampler.samples,
        ),
        "initial_boundary": (
            "admitted kt=2 stage-1 entry" if card == SWG_CARD
            else "certified day-0 state"),
    }
    if card == SWG_CARD and n_steps == 720:
        row["memory_1440_step_estimate_mib"] = extrapolate_memory(
            schedule, 720, 1440, sampler.baseline_mib, sampler.peak_mib)
        row["warm_reverse_1440_step_estimate_seconds"] = 2.0 * wall_s
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


def write_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("agreement", "resources"))
    parser.add_argument("--card", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--schedule", choices=SCHEDULES)
    parser.add_argument("--plant", choices=("broken_vjp",))
    parser.add_argument(
        "--transcendentals", choices=("libm", "native"), default="libm",
        help="libm is the certified CPU policy; GPU rows must select native",
    )
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "resources" and args.schedule is None:
        parser.error("resources mode requires --schedule")
    if args.mode == "agreement" and args.schedule is not None:
        parser.error("agreement mode does not accept --schedule")
    if args.mode == "resources" and args.plant is not None:
        parser.error("resources mode does not accept --plant")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    provenance = require_measurement_stamp(args.expect_commit)
    if args.mode == "agreement":
        row = measure_agreement(
            args.card, args.steps, plant=args.plant == "broken_vjp",
            transcendentals=args.transcendentals)
    else:
        row = measure_resources(
            args.card, args.steps, args.schedule,
            transcendentals=args.transcendentals)
    row["provenance"] = provenance
    row["argv"] = list(sys.argv if argv is None else argv)
    write_result(args.output, row)
    print(json.dumps(row, sort_keys=True), flush=True)
    return 0 if row["status"] == "CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
