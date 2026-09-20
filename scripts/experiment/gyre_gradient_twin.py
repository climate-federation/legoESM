#!/usr/bin/env python3
"""Identical-twin gradient recovery on the certified GYRE NEMO card.

Preregistration:
``docs/ocean/fidelity/PREREG_gyre_gradient_twin_2026-09-20.md``.

The reverse-mode/finite-difference comparison is a hard gate: no optimizer
update is made unless every requested coordinate passes at both registered
step sizes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

# This driver is a CPU experiment.  Set the platform before importing JAX;
# callers that already initialized a non-CPU backend are rejected below.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from legoesm.core.param_overrides import apply_param_overrides
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ml.loss import volume_weighted_mse
from legoesm.ml.training import TrainingConfig, create_optimizer
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_nemo_testcase_card,
    gyre_surface_forcings,
)
from legoesm.ocean.vertical import (
    nemo_qco_live_face_geometry_cgrid,
    nemo_qco_live_t_thickness,
    nemo_qco_resolved_mesh_operands,
)

CASE = "GYRE-zco"
DT_S = 14400.0
STEPS_PER_DAY = 6
FD_STEPS = (1.0e-3, 1.0e-4)
FD_RATIO_BOUNDS = (0.9999, 1.0001)
TRUTH = {"c_k": 0.1, "A_h": 1.0e5}
PARAM_CHOICES = ("c_k", "A_h", "both")
FIELD_NAMES = ("T", "S", "u", "v")


class TwinObservations(NamedTuple):
    """Scored prognostic fields plus eta for live-volume construction."""

    T: jax.Array
    S: jax.Array
    u: jax.Array
    v: jax.Array
    eta: jax.Array


class TwinReference(NamedTuple):
    """Fixed truth observations, volumes, and variance normalizers."""

    observations: TwinObservations
    T_volume: jax.Array
    S_volume: jax.Array
    u_volume: jax.Array
    v_volume: jax.Array
    T_variance: jax.Array
    S_variance: jax.Array
    u_variance: jax.Array
    v_variance: jax.Array


@dataclass(frozen=True)
class TwinContext:
    card: object
    model: LatLonCGridOceanModel
    initial_state: object


class InertParameterError(ValueError):
    """Raised when a selected parameter is absent from the reverse path."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def configure_runtime() -> None:
    """Enforce the preregistered CPU fp64/libm execution policy."""
    require(jax.default_backend() == "cpu", "GYRE gradient twin is CPU-only")
    require(bool(jax.config.jax_enable_x64), "JAX_ENABLE_X64=1 is required")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(
        get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
        "precision policy is not fp64/libm",
    )


def selected_parameters(param: str) -> tuple[str, ...]:
    if param == "both":
        return ("A_h", "c_k")
    if param not in ("A_h", "c_k"):
        raise ValueError(f"unknown parameter arm {param!r}")
    return (param,)


def build_context() -> TwinContext:
    """Build and scan-seed exactly the certified campaign card."""
    card = build_nemo_testcase_card(CASE)
    require(card.dt_s == DT_S, f"card dt {card.dt_s} != {DT_S}")
    config = card.recipe.model_config
    require(config.outer_integrator == "forward_euler", "GYRE card integrator changed")
    require(not config.polar_filter.use_polar_filter, "unexpected GYRE polar filter")
    require(not config.freeze_floor, "unexpected GYRE freeze floor")
    require(not config.ew_cyclic_overlap, "unexpected GYRE cyclic-overlap projection")
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, config)
    # ``nemo_ab3am4`` normally represents its first-window state as
    # ``bt_hist=None`` and writes a six-array tuple after step one.  A scan
    # carry cannot change treedef.  Give the carry an unreachable shape/dtype
    # placeholder; ``rollout_observations`` replaces it by None *inside* the
    # kt=1 cond branch, so the executed first step remains the campaign's exact
    # cold-start program while both cond branches return the tuple treedef.
    raw = card.recipe.initial_state
    uu_b = getattr(raw.uu_b, "data", raw.uu_b)
    vv_b = getattr(raw.vv_b, "data", raw.vv_b)
    require(uu_b is not None and vv_b is not None,
            "GYRE WS-RK3 state lacks prognostic uu_b/vv_b")
    bt_placeholder = (
        jnp.zeros_like(uu_b),
        jnp.zeros_like(uu_b),
        jnp.zeros_like(vv_b),
        jnp.zeros_like(vv_b),
        jnp.zeros_like(raw.eta.data),
        jnp.zeros_like(raw.eta.data),
    )
    initial = model.seed_scan_carry(
        raw._replace(bt_hist=bt_placeholder), card.dt_s)
    return TwinContext(card=card, model=model, initial_state=initial)


def config_with_parameters(base_config, values: Mapping[str, jax.Array]):
    """Splice the two supported leaves through the shared override helper."""
    unknown = set(values) - set(TRUTH)
    if unknown:
        raise ValueError(f"unsupported GYRE twin parameter(s): {sorted(unknown)}")
    config = base_config
    if "A_h" in values:
        lateral = apply_param_overrides(
            config.lateral_viscosity, {"A_h": values["A_h"]})
        config = apply_param_overrides(
            config, {"lateral_viscosity": lateral})
    if "c_k" in values:
        vmix = config.physics.vertical_mixing
        tke = apply_param_overrides(vmix.tke, {"c_k": values["c_k"]})
        vmix = apply_param_overrides(vmix, {"tke": tke})
        physics = apply_param_overrides(
            config.physics, {"vertical_mixing": vmix})
        config = apply_param_overrides(config, {"physics": physics})
    return config


def rollout_observations(
    context: TwinContext,
    config,
    n_steps: int,
    obs_steps: Sequence[int],
    initial_state=None,
) -> TwinObservations:
    """Run one checkpointed ``lax.scan`` and retain registered observations."""
    obs_steps = tuple(int(step) for step in obs_steps)
    if not obs_steps or min(obs_steps) < 1 or max(obs_steps) > n_steps:
        raise ValueError(f"observation steps {obs_steps} outside 1..{n_steps}")
    card, model = context.card, context.model
    scan_initial = context.initial_state if initial_state is None else initial_state

    def one_step(state, kt):
        freshwater, surface = gyre_surface_forcings(card, state, kt)
        new_state = model._step_impl(
            state,
            card.dt_s,
            freshwater=freshwater,
            surface_forcing=surface,
            config=config,
            _nemo_ab3am4_cold_start=(kt == 1),
        )
        fields = TwinObservations(
            new_state.T.data,
            new_state.S.data,
            new_state.u.data,
            new_state.v.data,
            new_state.eta.data,
        )
        return new_state, fields

    checkpointed_step = jax.checkpoint(one_step)
    _, trajectory = jax.lax.scan(
        checkpointed_step,
        scan_initial,
        jnp.arange(1, n_steps + 1, dtype=jnp.int32),
    )
    indices = jnp.asarray([step - 1 for step in obs_steps], dtype=jnp.int32)
    return jax.tree_util.tree_map(lambda field: field[indices], trajectory)


def _live_volumes(context: TwinContext, observations: TwinObservations):
    """Build T/U/V live control volumes at each truth observation."""
    card = context.card
    state = context.initial_state
    grid = card.recipe.grid
    z_coord = card.recipe.z_coord
    dtype = observations.T.dtype
    u_mask_3d, v_mask_3d = compute_face_masks_3d(z_coord.is_active, grid)
    operands = nemo_qco_resolved_mesh_operands(
        z_coord,
        grid,
        u_mask_3d,
        v_mask_3d,
        dtype,
        observations.T.shape[-1],
    )
    active_t = (
        jnp.asarray(z_coord.is_active, dtype=dtype)
        * jnp.asarray(state.land_mask.data[..., None], dtype=dtype)
    )
    active_u = jnp.asarray(u_mask_3d, dtype=dtype)
    active_v = jnp.asarray(v_mask_3d, dtype=dtype)
    area_t = jnp.asarray(operands.area_t, dtype=dtype)
    area_u = jnp.asarray(grid.dx_u * grid.dy_u, dtype=dtype)
    area_v = jnp.asarray(grid.dx_v * grid.dy_v, dtype=dtype)

    def at_eta(eta):
        e3t = nemo_qco_live_t_thickness(
            eta,
            state.H_bathy.data,
            z_coord,
            dtype,
            e3t_0=operands.e3t_0,
        )
        e3u, e3v, _, _ = nemo_qco_live_face_geometry_cgrid(
            eta,
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

    t_volume, u_volume, v_volume = jax.vmap(at_eta)(observations.eta)
    return t_volume, u_volume, v_volume


def build_reference(
    context: TwinContext,
    observations: TwinObservations,
) -> TwinReference:
    """Freeze truth volumes and eight nonzero truth-variance normalizers."""
    t_volume, u_volume, v_volume = _live_volumes(context, observations)
    volumes = {"T": t_volume, "S": t_volume, "u": u_volume, "v": v_volume}
    variances: dict[str, jax.Array] = {}
    for name in FIELD_NAMES:
        field = getattr(observations, name)
        volume = volumes[name]
        rows = []
        for obs_index in range(field.shape[0]):
            weight = volume[obs_index]
            mean = jnp.sum(field[obs_index] * weight) / jnp.sum(weight)
            rows.append(volume_weighted_mse(field[obs_index], mean, weight))
        variance = jnp.stack(rows)
        values = np.asarray(variance)
        require(
            bool(np.all(np.isfinite(values)) and np.all(values > 0.0)),
            f"{name} truth variance is not finite and positive: {values}",
        )
        variances[name] = variance
    return TwinReference(
        observations=observations,
        T_volume=t_volume,
        S_volume=t_volume,
        u_volume=u_volume,
        v_volume=v_volume,
        T_variance=variances["T"],
        S_variance=variances["S"],
        u_variance=variances["u"],
        v_variance=variances["v"],
    )


def make_loss(
    context: TwinContext,
    reference: TwinReference,
    trainable: Sequence[str],
    n_steps: int,
    obs_steps: Sequence[int],
    *,
    severed: Sequence[str] = (),
):
    """Return the registered normalized identical-twin loss.

    ``severed`` is a planted unit-test violation: that leaf is replaced by its
    base-card value rather than the trainable value, and the no-inert gate must
    reject it.
    """
    trainable = tuple(trainable)
    severed = frozenset(severed)
    if not set(severed).issubset(trainable):
        raise ValueError("severed leaves must be selected trainable leaves")
    base = context.card.recipe.model_config

    def loss(
        log_parameters: Mapping[str, jax.Array],
        path_activity: Mapping[str, jax.Array] | None = None,
    ):
        if set(log_parameters) != set(trainable):
            raise ValueError(
                f"loss expected {sorted(trainable)}, got {sorted(log_parameters)}"
            )
        if path_activity is not None and set(path_activity) != set(trainable):
            raise ValueError(
                "path_activity must have exactly the trainable parameter keys"
            )
        values = {}
        for name in trainable:
            truth_value = jnp.asarray(TRUTH[name], dtype=jnp.float64)
            if name in severed:
                activity = jnp.asarray(0.0, dtype=jnp.float64)
            elif path_activity is None:
                activity = jnp.asarray(1.0, dtype=jnp.float64)
            else:
                activity = jnp.asarray(path_activity[name], dtype=jnp.float64)
            # activity=0 is the planted path break: the forward config receives
            # truth and is exactly independent of this log leaf.  Keeping the
            # switch dynamic lets live and broken controls reuse one executable.
            values[name] = truth_value + activity * (
                jnp.exp(log_parameters[name]) - truth_value)
        # This call is intentionally inside the differentiated loss: its
        # replacement leaves are tracers, not values baked into a model object.
        config = config_with_parameters(base, values)
        candidate = rollout_observations(
            context, config, n_steps=n_steps, obs_steps=obs_steps)
        scores = []
        for name in FIELD_NAMES:
            candidate_field = getattr(candidate, name)
            truth_field = getattr(reference.observations, name)
            volume = getattr(reference, f"{name}_volume")
            variance = getattr(reference, f"{name}_variance")
            for obs_index in range(candidate_field.shape[0]):
                mse = volume_weighted_mse(
                    candidate_field[obs_index],
                    truth_field[obs_index],
                    volume[obs_index],
                )
                scores.append(mse / variance[obs_index])
        return jnp.mean(jnp.stack(scores))

    return loss


def start_log_parameters(trainable: Sequence[str]) -> dict[str, jax.Array]:
    return {
        name: jnp.log(jnp.asarray(2.0 * TRUTH[name], dtype=jnp.float64))
        for name in trainable
    }


def assert_no_inert(
    gradients: Mapping[str, jax.Array],
    selected: Sequence[str] | None = None,
) -> None:
    """Require every selected leaf to have a finite, exactly nonzero gradient."""
    names = tuple(gradients) if selected is None else tuple(selected)
    missing = [name for name in names if name not in gradients]
    if missing:
        raise InertParameterError(f"no-inert gate missing gradients: {missing}")
    dead, poisoned = [], []
    for name in names:
        value = np.asarray(gradients[name])
        if not np.all(np.isfinite(value)):
            poisoned.append(name)
        elif np.all(value == 0.0):
            dead.append(name)
    if dead or poisoned:
        pieces = []
        if dead:
            pieces.append(f"identically-zero gradient: {sorted(dead)}")
        if poisoned:
            pieces.append(f"non-finite gradient: {sorted(poisoned)}")
        raise InertParameterError("no-inert-parameters gate: " + "; ".join(pieces))


def finite_difference_rows(
    loss_evaluator,
    log_parameters: Mapping[str, jax.Array],
    gradients: Mapping[str, jax.Array],
    selected: Sequence[str],
    steps: Sequence[float] = FD_STEPS,
) -> list[dict]:
    """Evaluate the preregistered central differences and AD/FD ratios."""

    def finite_or_none(value: float) -> float | None:
        # Keep the machine receipt strict JSON.  ``None`` records an
        # undefined/non-finite numerical result without emitting the
        # non-standard JSON tokens NaN or Infinity.
        return value if math.isfinite(value) else None

    rows = []
    for name in selected:
        ad = float(np.asarray(gradients[name]))
        for relative_step in steps:
            plus = dict(log_parameters)
            minus = dict(log_parameters)
            h = jnp.asarray(relative_step, dtype=jnp.float64)
            plus[name] = plus[name] + h
            minus[name] = minus[name] - h
            f_plus = float(np.asarray(loss_evaluator(plus)))
            f_minus = float(np.asarray(loss_evaluator(minus)))
            fd = (f_plus - f_minus) / (2.0 * relative_step)
            ratio = ad / fd if fd != 0.0 else math.nan
            passed = bool(
                math.isfinite(ad)
                and math.isfinite(fd)
                and math.isfinite(ratio)
                and FD_RATIO_BOUNDS[0] <= ratio <= FD_RATIO_BOUNDS[1]
            )
            rows.append(
                {
                    "parameter": name,
                    "relative_log_step": relative_step,
                    "ad_gradient": finite_or_none(ad),
                    "finite_difference": finite_or_none(fd),
                    "ratio": finite_or_none(ratio),
                    "status": "CONFIRMED" if passed else "REFUTED",
                }
            )
    return rows


def _physical_values(log_parameters: Mapping[str, jax.Array]) -> dict[str, float]:
    return {
        name: float(np.asarray(jnp.exp(value)))
        for name, value in log_parameters.items()
    }


def _success(
    loss_value: float,
    start_loss: float,
    values: Mapping[str, float],
) -> bool:
    return bool(
        loss_value < 1.0e-6 * start_loss
        and all(abs(values[name] - TRUTH[name]) / TRUTH[name] < 1.0e-2
                for name in values)
    )


def _rss_mib() -> float:
    # Linux reports ru_maxrss in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def optimize(
    value_and_grad,
    start: Mapping[str, jax.Array],
    start_value: float,
    start_gradients: Mapping[str, jax.Array],
    start_wall_s: float,
    steps: int,
) -> dict:
    """Run at most ``steps`` shared-optimizer updates and record every state."""
    training_config = TrainingConfig(
        lr=0.1,
        warmup_steps=5,
        total_steps=steps,
        weight_decay=0.0,
        grad_clip_norm=1.0,
        optimizer="adam",
    )
    optimizer = create_optimizer(training_config)
    params = dict(start)
    opt_state = optimizer.init(params)
    value = start_value
    gradients = dict(start_gradients)
    curve = [
        {
            "step": 0,
            "loss": value,
            "loss_fraction": 1.0,
            "parameters": _physical_values(params),
            "gradient_wall_seconds": start_wall_s,
            "peak_rss_mib": _rss_mib(),
        }
    ]
    success_step = 0 if _success(value, start_value, curve[0]["parameters"]) else None
    for update_index in range(1, steps + 1):
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        params = optax.apply_updates(params, updates)
        started = time.perf_counter()
        value_device, gradients_device = value_and_grad(params)
        value_device.block_until_ready()
        wall_s = time.perf_counter() - started
        value = float(np.asarray(value_device))
        gradients = dict(gradients_device)
        assert_no_inert(gradients, tuple(params))
        values = _physical_values(params)
        curve.append(
            {
                "step": update_index,
                "loss": value,
                "loss_fraction": value / start_value,
                "parameters": values,
                "gradient_wall_seconds": wall_s,
                "peak_rss_mib": _rss_mib(),
            }
        )
        print(
            json.dumps(
                {
                    "step": update_index,
                    "loss": value,
                    "parameters": values,
                    "gradient_wall_s": wall_s,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if _success(value, start_value, values):
            success_step = update_index
            break
    gradient_costs = [row["gradient_wall_seconds"] for row in curve[1:]]
    return {
        "optimizer": {
            "name": "legoesm.ml.training.create_optimizer",
            "kind": "adam",
            "peak_learning_rate": 0.1,
            "warmup_steps": 5,
            "cosine_total_steps": steps,
            "gradient_clip_norm": 1.0,
            "weight_decay": 0.0,
        },
        "curve": curve,
        "success": success_step is not None,
        "success_step": success_step,
        "updates_executed": len(curve) - 1,
        "median_gradient_wall_seconds": (
            statistics.median(gradient_costs) if gradient_costs else start_wall_s
        ),
        "peak_rss_mib": _rss_mib(),
        "final_log_parameters": {
            name: float(np.asarray(value)) for name, value in params.items()
        },
    }


def loss_landscape(
    loss_evaluator,
    selected: Sequence[str],
) -> dict[str, list[dict]]:
    """Registered 11-point one-dimensional scans with other leaves at truth."""
    factors = np.exp(np.linspace(np.log(0.25), np.log(4.0), 11))
    landscape = {}
    for scanned in selected:
        rows = []
        for factor in factors:
            params = {
                name: jnp.log(jnp.asarray(TRUTH[name], dtype=jnp.float64))
                for name in selected
            }
            params[scanned] = jnp.log(
                jnp.asarray(TRUTH[scanned] * factor, dtype=jnp.float64))
            value = float(np.asarray(loss_evaluator(params)))
            rows.append(
                {
                    "factor_of_truth": float(factor),
                    "parameter": float(TRUTH[scanned] * factor),
                    "loss": value,
                }
            )
        landscape[scanned] = rows
    return landscape


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_year_harness():
    path = (
        _repo_root()
        / "scripts/validate/ocean_fidelity/testcases/"
        "nemo_testcase_l2_gyre_year_fromrest.py"
    )
    spec = importlib.util.spec_from_file_location("gyre_year_fromrest_twin", path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def campaign_identity(
    root: Path,
    truth: TwinObservations,
    days: int,
    n_steps: int,
) -> dict:
    """Compare scan truth against the existing campaign's own day snapshot."""
    require(days * STEPS_PER_DAY == n_steps, "days/step count mismatch")
    harness = _load_year_harness()
    campaign_root = root / "campaign_reference"
    harness.run_member(
        0,
        campaign_root,
        days=days,
        snap_steps=n_steps,
    )
    path = campaign_root / "lego_seed0" / f"day{days:03d}.npz"
    require(path.is_file(), f"campaign harness did not write {path}")
    with np.load(path) as dataset:
        campaign_t = np.asarray(dataset["T"], dtype=np.float64)
    twin_t = np.asarray(truth.T[-1], dtype=np.float64)
    unequal = int(np.count_nonzero(
        twin_t.view(np.uint64) != campaign_t.view(np.uint64)))
    return {
        "status": "CONFIRMED" if unequal == 0 else "REFUTED",
        "n_unequal": unequal,
        "n_values": int(twin_t.size),
        "campaign_snapshot": str(path),
        "twin_sha256": hashlib.sha256(twin_t.tobytes()).hexdigest(),
        "campaign_sha256": hashlib.sha256(campaign_t.tobytes()).hexdigest(),
    }


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=_repo_root(), text=True).strip()


def provenance(argv: Sequence[str]) -> dict:
    status = _git_output("status", "--porcelain")
    return {
        "git_sha": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status": status.splitlines(),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "precision_policy": repr(get_policy()),
        "python": sys.executable,
        "argv": list(argv),
    }


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def parse_obs_days(value: str) -> tuple[int, ...]:
    try:
        days = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("obs days must be comma-separated integers") from error
    if not days or any(day <= 0 for day in days) or tuple(sorted(set(days))) != days:
        raise argparse.ArgumentTypeError("obs days must be positive, unique, and increasing")
    return days


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--param", choices=PARAM_CHOICES, required=True)
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--obs-days", type=parse_obs_days, default=(5, 10))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.days <= 0 or args.steps <= 0 or args.steps > 100:
        parser.error("--days must be positive and --steps must be in 1..100")
    if max(args.obs_days) > args.days:
        parser.error("every observation day must be within --days")
    return args


def run(args: argparse.Namespace, argv: Sequence[str]) -> tuple[dict, int]:
    configure_runtime()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    trainable = selected_parameters(args.param)
    n_steps = args.days * STEPS_PER_DAY
    obs_steps = tuple(day * STEPS_PER_DAY for day in args.obs_days)
    summary = {
        "format": "gyre-gradient-twin-v1",
        "arm": args.param,
        "trainable": list(trainable),
        "truth": TRUTH,
        "start_factor": 2.0,
        "days": args.days,
        "steps_per_day": STEPS_PER_DAY,
        "rollout_steps": n_steps,
        "observation_days": list(args.obs_days),
        "observation_steps": list(obs_steps),
        "dt_s": DT_S,
        "loss": {
            "fields": list(FIELD_NAMES),
            "reducer": "legoesm.ml.loss.volume_weighted_mse",
            "weights": "truth-observation live T/U/V control volumes",
            "normalization": "per-field/per-time truth volume-weighted variance",
            "aggregate": "arithmetic mean of field-time normalized scores",
        },
        "provenance": provenance(argv),
        "peak_rss_mib": _rss_mib(),
    }
    context = build_context()
    base = context.card.recipe.model_config
    require(float(base.physics.vertical_mixing.tke.c_k) == TRUTH["c_k"],
            "card c_k truth changed")
    require(float(base.lateral_viscosity.A_h) == TRUTH["A_h"],
            "card A_h truth changed")

    truth_config = config_with_parameters(
        base, {name: jnp.asarray(value, dtype=jnp.float64)
               for name, value in TRUTH.items()})
    truth_rollout = jax.jit(
        lambda initial: rollout_observations(
            context,
            truth_config,
            n_steps=n_steps,
            obs_steps=obs_steps,
            initial_state=initial,
        ))
    truth_started = time.perf_counter()
    truth = truth_rollout(context.initial_state)
    truth.T.block_until_ready()
    summary["truth_wall_seconds"] = time.perf_counter() - truth_started
    truth_dir = root / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    for index, day in enumerate(args.obs_days):
        np.savez(
            truth_dir / f"day{day:03d}.npz",
            T=np.asarray(truth.T[index]),
            S=np.asarray(truth.S[index]),
            u=np.asarray(truth.u[index]),
            v=np.asarray(truth.v[index]),
            eta=np.asarray(truth.eta[index]),
        )

    if args.days == 10 and args.obs_days[-1] == 10:
        summary["P5_campaign_identity"] = campaign_identity(
            root, truth, args.days, n_steps)
    else:
        summary["P5_campaign_identity"] = {
            "status": "UNMEASURED",
            "reason": "P5 is registered specifically at day 10",
        }
    if summary["P5_campaign_identity"]["status"] != "CONFIRMED":
        summary["overall_status"] = "STOPPED_P5"
        summary["peak_rss_mib"] = _rss_mib()
        return summary, 2

    reference = build_reference(context, truth)
    summary["truth_variances"] = {
        name: np.asarray(getattr(reference, f"{name}_variance")).tolist()
        for name in FIELD_NAMES
    }
    loss = make_loss(
        context,
        reference,
        trainable,
        n_steps=n_steps,
        obs_steps=obs_steps,
    )
    start = start_log_parameters(trainable)

    eager_started = time.perf_counter()
    eager_device = loss(start)
    eager_device.block_until_ready()
    eager_value = float(np.asarray(eager_device))
    eager_wall = time.perf_counter() - eager_started
    jitted_loss = jax.jit(loss)
    jit_started = time.perf_counter()
    jit_device = jitted_loss(start)
    jit_device.block_until_ready()
    jit_value = float(np.asarray(jit_device))
    jit_wall = time.perf_counter() - jit_started
    parity = abs(jit_value - eager_value) / abs(eager_value) if eager_value else math.inf
    summary["P4_jit_parity"] = {
        "status": "CONFIRMED" if math.isfinite(parity) and parity <= 1.0e-12 else "REFUTED",
        "eager_loss": eager_value,
        "jitted_loss": jit_value,
        "relative_difference": parity,
        "eager_wall_seconds": eager_wall,
        "first_jit_wall_seconds": jit_wall,
    }
    if summary["P4_jit_parity"]["status"] != "CONFIRMED":
        summary["overall_status"] = "STOPPED_P4"
        summary["peak_rss_mib"] = _rss_mib()
        return summary, 2

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    gradient_started = time.perf_counter()
    start_value_device, gradients_device = value_and_grad(start)
    start_value_device.block_until_ready()
    start_gradient_wall = time.perf_counter() - gradient_started
    start_value = float(np.asarray(start_value_device))
    gradients = dict(gradients_device)
    # Always measure the registered finite differences, including when AD is
    # poisoned.  A non-finite AD value then yields an explicit null ratio and
    # REFUTED row instead of suppressing the comparison behind no-inert.
    fd_rows = finite_difference_rows(
        jitted_loss, start, gradients, trainable, steps=FD_STEPS)
    fd_passed = all(row["status"] == "CONFIRMED" for row in fd_rows)
    summary["P1_adjoint"] = {
        "status": "CONFIRMED" if fd_passed else "REFUTED",
        "ratio_bounds": list(FD_RATIO_BOUNDS),
        "rows": fd_rows,
    }
    try:
        assert_no_inert(gradients, trainable)
        summary["assert_no_inert"] = {
            "status": "CONFIRMED",
            "gradients": {name: float(np.asarray(gradients[name])) for name in trainable},
        }
    except InertParameterError as error:
        summary["assert_no_inert"] = {"status": "REFUTED", "reason": str(error)}
        summary["overall_status"] = "STOPPED_INERT"
        summary["peak_rss_mib"] = _rss_mib()
        return summary, 2

    if not fd_passed:
        summary["overall_status"] = "STOPPED_P1"
        summary["localization"] = (
            "Reverse-mode defect requires operation-level localization; no "
            "optimizer update was made."
        )
        summary["peak_rss_mib"] = _rss_mib()
        return summary, 2

    recovery = optimize(
        value_and_grad,
        start,
        start_value,
        gradients,
        start_gradient_wall,
        args.steps,
    )
    summary["recovery"] = recovery
    if not recovery["success"]:
        summary["identifiability_scan"] = loss_landscape(jitted_loss, trainable)
        summary["optimizer_retune"] = {
            "performed": False,
            "reason": "registered initial attempt exhausted; landscape reported without tuning",
        }
    summary["recovery_status"] = "CONFIRMED" if recovery["success"] else "REFUTED"
    summary["overall_status"] = (
        "CONFIRMED" if recovery["success"] else "RECOVERY_STALLED"
    )
    summary["peak_rss_mib"] = _rss_mib()
    return summary, 0 if recovery["success"] else 3


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    summary_path = args.root.resolve() / f"gyre_gradient_twin_{args.param}.json"
    try:
        summary, exit_code = run(args, actual_argv)
    except Exception as error:  # fail closed while retaining a machine receipt
        summary = {
            "format": "gyre-gradient-twin-v1",
            "arm": args.param,
            "overall_status": "ERROR",
            "error_type": type(error).__name__,
            "error": str(error),
            "argv": actual_argv,
            "peak_rss_mib": _rss_mib(),
        }
        write_summary(summary_path, summary)
        raise
    write_summary(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), "status": summary["overall_status"]}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
