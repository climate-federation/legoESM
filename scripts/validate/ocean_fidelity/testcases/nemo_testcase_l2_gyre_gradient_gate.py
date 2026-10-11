#!/usr/bin/env python3
"""Reverse/forward/FD and adjoint-resource gate for certified GYRE-zco.

The numerical protocol is frozen in
``nemo_testcases_l2_gyre_gradgate_round1_prereg.md``.  This module deliberately
contains no physics repair: a poisoned adjoint is an exit-1 measurement.
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
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
from legoesm.ocean.vertical import (
    nemo_qco_live_face_geometry_cgrid,
    nemo_qco_live_t_thickness,
    nemo_qco_resolved_mesh_operands,
)
from legoesm.training.checkpoint_schedule import checkpointed_loop

CASE = "GYRE-zco"
DT_S = 14400.0
REVFWD_TOL = 1.0e-9
FD_TOL = 0.05
FD_STEP = 1.0e-3
PERTURBATION_K = 1.0e-3
SCHEDULES = ("none", "uniform", "binomial")
FIELDS = ("T", "S", "u", "v")


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
    checkpointed_step: object


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def configure_runtime() -> None:
    """Select and verify the registered fp64/libm compute policy."""
    require(bool(jax.config.jax_enable_x64), "JAX_ENABLE_X64=1 is required")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(
        get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
        "precision policy is not fp64/libm",
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


def build_context() -> GYREContext:
    """Build the certified card and prime only production step caches."""
    configure_runtime()
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
    def production_step(state, freshwater, surface):
        return model._step_jitted(state, card.dt_s, freshwater, surface)

    # Reuse Decision 49 from the GYRE gradient-twin lane: preserve a compiled
    # boundary around every production step and rematerialize that whole step
    # in reverse mode.  This remains a Python loop (never a scan).
    checkpointed_step = jax.jit(jax.checkpoint(production_step))
    return GYREContext(
        card, model, initial, _load_surface_forcings(), checkpointed_step)


def _score_state(state) -> ScoredState:
    return ScoredState(
        state.T.data, state.S.data, state.u.data, state.v.data, state.eta.data)


def production_rollout(context: GYREContext, d_t0, n_steps: int):
    """Campaign-order Python loop; every step calls production ``_step_jitted``."""
    state = context.initial_state._replace(
        T=context.initial_state.T.replace(
            data=context.initial_state.T.data + d_t0))
    for kt in range(1, n_steps + 1):
        freshwater, surface = context.surface_forcings(context.card, state, kt)
        state = context.model._step_jitted(
            state,
            context.card.dt_s,
            freshwater,
            surface,
        )
    return state


def gradient_rollout(context: GYREContext, d_t0, n_steps: int):
    """Twin-Decision-49 per-step remat around production ``_step_jitted``."""
    state = context.initial_state._replace(
        T=context.initial_state.T.replace(
            data=context.initial_state.T.data + d_t0))
    for kt in range(1, n_steps + 1):
        freshwater, surface = context.surface_forcings(context.card, state, kt)
        state = context.checkpointed_step(state, freshwater, surface)
    return state


def scheduled_rollout(context: GYREContext, d_t0, n_steps: int, schedule: str):
    """Run the shared component-agnostic checkpoint schedule over production steps."""
    state = context.initial_state._replace(
        T=context.initial_state.T.replace(
            data=context.initial_state.T.data + d_t0))

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
    rollout = gradient_rollout if schedule is None else scheduled_rollout

    def loss(d_t0):
        if schedule is None:
            state = rollout(context, d_t0, n_steps)
        else:
            state = rollout(context, d_t0, n_steps, schedule)
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
    zero = jnp.asarray(0.0, dtype=base.dtype)

    def line(alpha):
        return loss(base + alpha * direction)

    reverse_line = (
        (lambda alpha: line(broken_vjp_identity(alpha))) if plant else line)
    reverse = float(np.asarray(jax.grad(reverse_line)(zero)))
    _, forward_value = jax.jvp(line, (zero,), (jnp.asarray(1.0, base.dtype),))
    forward = float(np.asarray(forward_value))
    plus = float(np.asarray(line(jnp.asarray(FD_STEP, base.dtype))))
    minus = float(np.asarray(line(jnp.asarray(-FD_STEP, base.dtype))))
    finite_difference = (plus - minus) / (2.0 * FD_STEP)
    return classify_agreement(reverse, forward, finite_difference)


def measure_agreement(n_steps: int, *, plant=False):
    """Build and evaluate one certified-card agreement row."""
    context = build_context()
    zero = jnp.zeros_like(context.initial_state.T.data)
    truth_state = production_rollout(context, zero, n_steps)
    jax.block_until_ready(truth_state)
    gradient_truth = gradient_rollout(context, zero, n_steps)
    jax.block_until_ready(gradient_truth)
    require(states_bitwise_equal(truth_state, gradient_truth),
            "checkpointed per-step primal differs from production")
    reference = build_reference(context, truth_state)
    loss = make_loss(context, reference, n_steps)
    base, direction = perturbations(context)
    row = classify_agreement(math.nan, math.nan, math.nan)
    try:
        row = directional_derivatives(loss, base, direction, plant=plant)
    except (TypeError, ValueError, RuntimeError) as error:
        row["error"] = f"{type(error).__name__}: {error}"
    row.update({
        "case": CASE,
        "steps": n_steps,
        "model_seconds": n_steps * DT_S,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "precision_policy": repr(get_policy()),
        "state_dtype": str(context.initial_state.T.data.dtype),
        "geometry_dtype": str(context.card.recipe.z_coord.dz_ref.dtype),
        "plant": "broken_vjp" if plant else None,
        "production_entry": "LatLonCGridOceanModel._step_jitted",
        "composition": "per-step Python loop",
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


def measure_resources(n_steps: int, schedule: str):
    """Warm once, then measure one blocked reverse execution and peak memory."""
    context = build_context()
    zero = jnp.zeros_like(context.initial_state.T.data)
    truth_state = production_rollout(context, zero, n_steps)
    jax.block_until_ready(truth_state)
    reference = build_reference(context, truth_state)
    scheduled_truth = scheduled_rollout(context, zero, n_steps, schedule)
    jax.block_until_ready(scheduled_truth)
    require(states_bitwise_equal(truth_state, scheduled_truth),
            f"{schedule} schedule primal differs from per-step production")
    loss = make_loss(context, reference, n_steps, schedule=schedule)
    base, direction = perturbations(context)

    def line(alpha):
        return loss(base + alpha * direction)

    reverse = jax.value_and_grad(line)
    zero_alpha = jnp.asarray(0.0, dtype=base.dtype)

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
    return {
        "case": CASE,
        "steps": n_steps,
        "model_seconds": n_steps * DT_S,
        "schedule": schedule,
        "backend": backend,
        "device": str(jax.devices()[0]),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "precision_policy": repr(get_policy()),
        "state_dtype": str(context.initial_state.T.data.dtype),
        "geometry_dtype": str(context.card.recipe.z_coord.dz_ref.dtype),
        "production_entry": "LatLonCGridOceanModel._step_jitted",
        "schedule_primal_bitwise_production": True,
        "cold_compile_and_reverse_wall_seconds": cold_wall_s,
        "warm_reverse_wall_seconds": wall_s,
        "loss": float(np.asarray(value)),
        "gradient_finite": bool(np.all(np.isfinite(gradient_array))),
        "gradient_nonzero": bool(np.any(gradient_array != 0.0)),
        "memory_kind": "host_rss" if backend == "cpu" else "gpu1_nvidia_smi",
        "memory_baseline_mib": sampler.baseline_mib,
        "memory_peak_mib": sampler.peak_mib,
        "memory_increment_mib": sampler.peak_mib - sampler.baseline_mib,
        "memory_samples": sampler.samples,
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


def write_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("agreement", "resources"))
    parser.add_argument("--steps", type=int, choices=(1, 6), required=True)
    parser.add_argument("--schedule", choices=SCHEDULES)
    parser.add_argument("--plant", choices=("broken_vjp",))
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "resources" and args.schedule is None:
        parser.error("resources mode requires --schedule")
    if args.mode == "agreement" and args.schedule is not None:
        parser.error("agreement mode does not accept --schedule")
    if args.mode == "resources" and args.plant is not None:
        parser.error("resources mode does not accept --plant")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    provenance = require_measurement_stamp(args.expect_commit)
    if args.mode == "agreement":
        row = measure_agreement(
            args.steps, plant=args.plant == "broken_vjp")
    else:
        row = measure_resources(args.steps, args.schedule)
    row["provenance"] = provenance
    row["argv"] = list(sys.argv if argv is None else argv)
    write_result(args.output, row)
    print(json.dumps(row, sort_keys=True), flush=True)
    if args.mode == "agreement":
        return 0 if row["status"] == "CONFIRMED" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
