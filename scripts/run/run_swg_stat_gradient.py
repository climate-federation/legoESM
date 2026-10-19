#!/usr/bin/env python3
"""Arm E: gradients of a climate-style statistic loss of the SWG online closure.

Where does gradient training of a closure stop working when the loss is a
CLIMATE statistic of a long free window (Perezhogin, Adcroft & Zanna 2026
used ensemble Kalman inversion for the same kind of loss) instead of the
Eq. 7 trajectory loss?  ``measure`` computes, at one checkpoint, the loss and
its gradient from each of several truth start states; ``train`` repeats Adam
steps on the mean gradient.

Each window starts from a stored truth state (``SWGDatasetContract.start_state``,
as every training window) and runs ``--window-days`` model days with the
closure.  The daily end-of-day snapshots (the free-run sampling the held-out
scorer admits) are reduced to statistics and scored against the truth's
statistics over the SAME absolute window:

``transport_eke``  the time-mean native face transport map (U and V faces
    together) and the EKE map (cell-centred velocity anomaly about the window
    mean), each as NRMSE squared with the scorer's definitions
    (``nemo_testcase_l1_swg_online_evaluation.score_statistics``:
    ``mean_face_transport_rmse_m3_s`` over the wet faces, ``wet_eke_map_nrmse``
    over the wet T points; the transport RMSE is normalised by the truth's
    RMS transport), summed with equal weight.
``phi_mean_std``  the time-mean and the temporal standard deviation (ddof 0)
    maps of the lid potential over g, ``phi/g`` [m] (rigid-lid decks only),
    each as NRMSE squared over the wet T points, summed with equal weight.
    ``phi`` of a state is ``rigid_lid_start_state``'s stage-1 potential on
    both the model and the truth side.

The closure is the checkpoint's network with its potential multiplied by a
scalar ``gamma`` (initially 1.0; see ``ScaledClosure``).  ``--trainable``
selects ``gamma``; ``gamma`` and the last convolution's weights
(``gamma_last_layer``); or every network leaf and ``gamma`` (``all``).  The
model day is ``build_segment_fn`` (one day, one record) with per-step
rematerialisation; days are a ``lax.scan`` that carries the model state and a
streaming (Welford) mean and sum of squared deviations of each reduced field,
so no trajectory is stored: the backward pass keeps one model state per step
(about 3 MB per model day in float64 on the 84 x 84 deck).
``--precision fp32`` traces and differentiates in float32 from float64
masters, as the training driver does.

``free-run`` free-runs one saved ``train`` step with the training driver's own
free-run code (``free_run``, the configuration's ``free_run`` block, device and
output layout): the step's ``effective_network`` is written as a plain network
checkpoint in the driver's format (``--network-output``) and that file, loaded
back as the driver loads a checkpoint, is the closure the free run names.

No command has a default for any scientific choice; the output directory
holds ``manifest.json`` (git SHA, every argument, the start days) beside the
results.
"""

import argparse
import contextlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from legoesm.ml.checkpoint_io import load_checkpoint_or_fail
from legoesm.ml.training import save_checkpoint
from legoesm.ocean.diagnostics import velocity_to_cell_centre
from legoesm.ocean.fidelity.swg_coarse_graining import (
    build_jax_native_transport_diagnostic,
    cgrid_geometry,
)
from legoesm.training.swg_online import (
    HOURS_PER_DAY,
    SWGDatasetContract,
    SWGOptimizerSettings,
    SWGScalarPotentialCNN,
    build_closure_geometry,
    build_segment_fn,
    coarse_state_from_native,
    create_swg_optimizer,
    free_run_closure,
    require_rigid_lid_zero_ssh,
    rigid_lid_start_state,
    start_state_from_native,
    stored_state_at,
    window_steps,
)

SCHEMA = "legoesm-swg-stat-gradient-v1"
STATISTICS = ("transport_eke", "phi_mean_std")
SUBSETS = ("gamma", "gamma_last_layer", "all")
PRECISIONS = ("fp64", "fp32")
FD_STEPS = (1.0e-2, 1.0e-3, 1.0e-4)
GAMMA_INITIAL = 1.0
# The key reaches the closure only through ``noise_scale * normal(key)`` with
# noise_scale 0, so its value never enters a result.
CLOSURE_KEY_SEED = 0


def _load_driver():
    """The training driver, loaded by path: its card, device, precision and
    network-spec helpers are the ones every SWG run uses."""
    path = Path(__file__).with_name("run_swg_online_training.py")
    spec = importlib.util.spec_from_file_location(
        "run_swg_online_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _load_driver()


# ------------------------------------------------------------------- closure

class ScaledClosure(eqx.Module):
    """The loaded network and a scalar ``gamma`` on its potential.

    ``effective_network`` is the network whose potential is ``gamma`` times
    the loaded one's: the final convolution has no bias, so the potential is
    linear in its weights and ``gamma * weights`` scales it exactly.  The
    scaling happens once, outside the stepper's scan, so at ``gamma == 1.0``
    the stepper receives the loaded network's own leaves (``1.0 * w == w``)
    and runs the plain closure's program bit for bit; multiplying the
    potential inside the step instead changes XLA's fusion and moves the
    trajectory by about one unit in the last place.
    """

    network: SWGScalarPotentialCNN
    gamma: jax.Array

    def effective_network(self) -> SWGScalarPotentialCNN:
        return eqx.tree_at(lambda n: n.layers[-1].weight, self.network,
                           replace_fn=lambda weight: self.gamma * weight)


def trainable_spec(model: ScaledClosure, subset: str):
    """Boolean pytree of ``model``: True on the leaves ``subset`` trains."""
    if subset not in SUBSETS:
        raise ValueError(f"trainable must be one of {SUBSETS}, got {subset!r}")
    if subset == "all":
        return jax.tree.map(eqx.is_inexact_array, model)
    selected = (lambda m: m.gamma) if subset == "gamma" else (
        lambda m: (m.gamma, m.network.layers[-1].weight))
    return eqx.tree_at(selected, jax.tree.map(lambda _: False, model),
                       replace_fn=lambda _: True)


def adam_step(optimizer, opt_state, trainable, frozen, grads):
    """One optimiser step on the trainable leaves; frozen leaves untouched."""
    updates, opt_state = optimizer.update(grads, opt_state, trainable)
    return eqx.combine(eqx.apply_updates(trainable, updates), frozen), opt_state


# ---------------------------------------------------------------- statistics

class StatisticGeometry(NamedTuple):
    """The deck's C-grid masks and transport diagnostic, float64 constants."""

    masks: dict
    transports: object


def statistic_geometry(namelist) -> StatisticGeometry:
    """Built with x64 on (``cgrid_geometry`` builds and validates a float64
    card), so a float32 program takes it from outside its trace."""
    mask = cgrid_geometry(namelist)["mask"]
    return StatisticGeometry(
        {kind: jnp.asarray(mask[kind]) for kind in "TUV"},
        build_jax_native_transport_diagnostic(namelist))


class Statistic:
    """One statistic set on one card: per-state fields, a streaming
    reduction of the daily fields, the maps and the NRMSE-squared loss."""

    def __init__(self, card, name: str, geometry: StatisticGeometry):
        if name not in STATISTICS:
            raise ValueError(f"statistic must be one of {STATISTICS}, got {name!r}")
        self.name = name
        self.card = card
        self.lid = card.recipe.model_config.swe_surface == "rigid_lid"
        if name == "phi_mean_std" and not self.lid:
            raise ValueError("phi_mean_std needs a rigid-lid card")
        self.masks = geometry.masks
        self.transports = geometry.transports
        self.g = card.recipe.model_config.constants.g

    def fields(self, state) -> dict:
        """The reduced fields of one state (native eta, u, v and the lid
        potential), traceable."""
        eta = state.eta.data
        u = state.u.data[:, 1:, 0]
        v = state.v.data[1:, :, 0]
        if self.name == "phi_mean_std":
            return {"phi": rigid_lid_start_state(
                self.card, state).rigid_lid_phi / self.g}
        transport_u, transport_v = self.transports(eta, u, v)
        # The scorer's ``_cell_centred``: masked faces, a zero west/south
        # face, then the two-face average to the T points.
        u_faces = jnp.where(self.masks["U"], u, 0.0)
        v_faces = jnp.where(self.masks["V"], v, 0.0)
        u_c, v_c = velocity_to_cell_centre(
            jnp.concatenate([jnp.zeros_like(u_faces[:, :1]), u_faces], axis=1),
            jnp.concatenate([jnp.zeros_like(v_faces[:1, :]), v_faces], axis=0))
        return {"transport_u": transport_u, "transport_v": transport_v,
                "u_c": u_c, "v_c": v_c}

    @staticmethod
    def accumulator(fields: dict):
        """Streaming state after one frame: (count, mean, sum sq. deviation)."""
        return (jnp.ones((), dtype=jax.tree.leaves(fields)[0].dtype), fields,
                jax.tree.map(jnp.zeros_like, fields))

    @staticmethod
    def update(accumulator, fields: dict):
        """Welford update with one more frame (stable, differentiable)."""
        count, mean, squares = accumulator
        count = count + 1.0
        delta = jax.tree.map(lambda x, m: x - m, fields, mean)
        mean = jax.tree.map(lambda m, d: m + d / count, mean, delta)
        squares = jax.tree.map(
            lambda s, d, x, m: s + d * (x - m), squares, delta, fields, mean)
        return count, mean, squares

    def maps(self, accumulator) -> dict:
        """The statistic maps of a window (population statistics, as the
        scorer's mean over the window's days)."""
        count, mean, squares = accumulator
        if self.name == "phi_mean_std":
            wet = self.masks["T"]
            return {"phi_mean": mean["phi"],
                    "phi_std": jnp.sqrt(jnp.where(
                        wet, squares["phi"] / count, 1.0))}
        return {"transport_u": mean["transport_u"],
                "transport_v": mean["transport_v"],
                "eke": 0.5 * (squares["u_c"] + squares["v_c"]) / count}

    def groups(self) -> list:
        """Equal-weight loss terms, each a tuple of (map name, mask kind)."""
        if self.name == "phi_mean_std":
            return [(("phi_mean", "T"),), (("phi_std", "T"),)]
        return [(("transport_u", "U"), ("transport_v", "V")), (("eke", "T"),)]

    def terms(self, candidate, reference) -> list:
        """NRMSE squared of each loss term: sum (c - r)^2 / sum r^2 over the
        term's wet points (both face kinds together for the transport)."""
        cand, ref = self.maps(candidate), self.maps(reference)
        terms = []
        for group in self.groups():
            numerator = sum(jnp.sum(jnp.where(
                self.masks[kind], (cand[name] - ref[name]) ** 2, 0.0))
                for name, kind in group)
            denominator = sum(jnp.sum(jnp.where(
                self.masks[kind], ref[name] ** 2, 0.0)) for name, kind in group)
            terms.append(numerator / denominator)
        return terms

    def loss(self, candidate, reference):
        return sum(self.terms(candidate, reference))


def truth_accumulator(card, statistic: str, frames):
    """The statistic's streaming reduction of truth ``frames`` (daily native
    (ssh, u, v) stacks, float64), computed by the model side's own code."""
    stat = Statistic(card, statistic, statistic_geometry(card.resolved_namelist))
    require_rigid_lid_zero_ssh(card, frames[0])
    template = card.recipe.initial_state

    @jax.jit
    def reduce(ssh, u, v):
        per_frame = jax.lax.map(
            lambda frame: stat.fields(coarse_state_from_native(
                template, *frame)), (ssh, u, v))
        first = jax.tree.map(lambda x: x[0], per_frame)
        rest = jax.tree.map(lambda x: x[1:], per_frame)
        accumulator, _ = jax.lax.scan(
            lambda acc, fields: (stat.update(acc, fields), None),
            stat.accumulator(first), rest)
        return accumulator

    return reduce(*(jnp.asarray(value) for value in frames))


# ------------------------------------------------------------------ programs

def _single_card(card):
    return card._replace(recipe=card.recipe._replace(
        grid=driver._single(card.recipe.grid),
        land_mask=driver._single(card.recipe.land_mask),
        initial_state=driver._single(card.recipe.initial_state)))


def build_programs(card, geometry, *, days: int, statistic: str,
                   precision: str):
    """Jitted ``(value_and_grad, loss)`` of the window statistic loss.

    Both take ``(trainable, frozen, start_state, truth_accumulator)`` as
    float64 masters and return float64; ``precision="fp32"`` traces and runs
    them with x64 off, as the training driver's batch functions do.
    ``value_and_grad`` differentiates ``trainable`` only.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    if type(days) is not int or days < 2:
        raise ValueError("window_days must be an integer >= 2 (a standard "
                         "deviation needs two daily frames)")
    single = precision == "fp32"
    constants = statistic_geometry(card.resolved_namelist)
    with driver._single_precision_trace() if single else contextlib.nullcontext():
        if single:
            card, geometry = _single_card(card), driver._single(geometry)
        raw, _ = build_segment_fn(
            card, geometry, window_days=1, target_interval_hours=24,
            checkpoint=True)
        stat = Statistic(card, statistic, constants)
    steps = window_steps(card, 1, 24)
    shape = tuple(geometry.tmask.shape)
    dtype = jnp.float32 if single else jnp.float64

    def window(network, start_state, key):
        noise = jnp.zeros((steps, 3) + shape, dtype=dtype)

        def one_day(state):
            state, _ = raw(network, state, noise, key, jnp.asarray(0.0))
            return state, stat.fields(state)

        # The first day runs outside the scan: ``raw`` adds the wind-stress
        # carry to a start state that has none, which a scan carry cannot.
        state, fields = one_day(start_state)

        def body(carry, _):
            state, accumulator = carry
            state, fields = one_day(state)
            return (state, stat.update(accumulator, fields)), None

        (_, accumulator), _ = jax.lax.scan(
            body, (state, stat.accumulator(fields)), None, length=days - 1)
        return accumulator

    key = jax.random.PRNGKey(CLOSURE_KEY_SEED)

    def loss(trainable, frozen, start_state, truth):
        network = eqx.combine(trainable, frozen).effective_network()
        accumulator = window(network, start_state, key)
        return stat.loss(accumulator, truth)

    value_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(loss))
    loss_fn = eqx.filter_jit(loss)
    if single:
        def at_single(function):
            def run(*arguments):
                with driver._single_precision_trace():
                    result = function(*(driver._single(a) for a in arguments))
                return jax.tree.map(lambda x: x.astype(jnp.float64), result)
            return run
        value_and_grad, loss_fn = at_single(value_and_grad), at_single(loss_fn)
    return value_and_grad, loss_fn


# ------------------------------------------------------------------ the data

def check_start_days(contract, days: int, train, heldout) -> None:
    """Refuse start days whose windows leave their split or overlap.

    Training starts must lie in the contract's train split, held-out starts
    in its validation split; each list is strictly increasing with starts at
    least ``days`` apart, and every window lies inside its split.
    """
    bounds = {name: (first, last) for name, first, last in contract.splits}
    for label, split, starts in (("train", "train", train),
                                 ("held-out", "validation", heldout)):
        if not starts or any(type(day) is not int for day in starts):
            raise ValueError(f"{label} start days must be a non-empty integer list")
        if any(b - a < days for a, b in zip(starts[:-1], starts[1:], strict=True)):
            raise ValueError(
                f"{label} start days must increase with spacing >= {days} days")
        first, last = bounds[split]
        for day in starts:
            if HOURS_PER_DAY * day < first or HOURS_PER_DAY * (day + days) > last:
                raise ValueError(
                    f"{label} window [{day}, {day + days}] days leaves the "
                    f"{split} split [{first // HOURS_PER_DAY}, "
                    f"{last // HOURS_PER_DAY}]")


def truth_frames(contract, start_day: int, days: int):
    """The stored (ssh, u, v) of the contract's operator at the end of each of
    the ``days`` days after ``start_day``: stacks of shape (days, y, x)."""
    if contract.target_filter_sigma_cells != 1.0:
        raise ValueError(
            "truth frames are read as stored; a widened target filter "
            "(target_filter_sigma_cells > 1) is not supported here")
    frames = [stored_state_at(
        contract.sources, contract.objective,
        HOURS_PER_DAY * (start_day + k))[0] for k in range(1, days + 1)]
    return tuple(np.stack([frame[i] for frame in frames]) for i in range(3))


class Window(NamedTuple):
    start_day: int
    start_state: object
    truth: object


def load_windows(contract, card, statistic: str, days: int, start_days):
    """Start state and truth statistics of each start day (float64)."""
    windows = []
    for day in start_days:
        (ssh, u, v), _ = contract.start_state(HOURS_PER_DAY * day)
        windows.append(Window(day, start_state_from_native(card, ssh, u, v),
                              truth_accumulator(card, statistic,
                                                truth_frames(contract, day, days))))
    return windows


def load_model(config: dict, card, geometry, checkpoint: Path) -> ScaledClosure:
    """The checkpoint's network (loaded as the driver does) with gamma 1."""
    spec, _ = driver.network_spec(config, card, geometry)
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(0))
    optimizer = create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))
    network, _ = load_checkpoint_or_fail(
        checkpoint, (network, optimizer.init(
            eqx.filter(network, eqx.is_inexact_array))),
        what="SWG closure and optimiser")
    return ScaledClosure(network, jnp.asarray(GAMMA_INITIAL))


# ----------------------------------------------------------------- the steps

def window_values(programs, trainable, frozen, windows, *, gradients: bool):
    """Per-window losses (and flat gradients) at the given parameters."""
    value_and_grad, loss_fn = programs
    losses, grads = [], []
    for window in windows:
        if gradients:
            loss, grad = value_and_grad(
                trainable, frozen, window.start_state, window.truth)
            grads.append(np.asarray(ravel_pytree(grad)[0]))
        else:
            loss = loss_fn(trainable, frozen, window.start_state, window.truth)
        losses.append(float(loss))
    return np.asarray(losses), (np.stack(grads) if gradients else None)


def evaluate(setup, trainable, frozen):
    """Per-start training losses and gradients, and held-out losses.

    The gradient is of the training starts only; the held-out starts are
    evaluated without differentiation.
    """
    losses, grads = window_values(
        setup.programs, trainable, frozen, setup.train, gradients=True)
    heldout, _ = window_values(
        setup.programs, trainable, frozen, setup.heldout, gradients=False)
    return losses, grads, heldout


def parameter_names(trainable) -> list:
    return [{"name": jax.tree_util.keystr(path), "size": int(leaf.size)}
            for path, leaf in jax.tree_util.tree_leaves_with_path(
                eqx.filter(trainable, eqx.is_inexact_array))]


def _cosines(grads: np.ndarray, mean: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(grads, axis=1) * np.linalg.norm(mean)
    return np.where(norms > 0.0, grads @ mean / np.where(norms > 0.0, norms, 1.0),
                    np.nan)


def measure(setup, *, lr: float, weight_decay: float) -> tuple[dict, np.ndarray]:
    """Per-start loss, gradient and cosines; gamma finite differences; one
    Adam step on the mean gradient and the losses it leaves."""
    model, programs = setup.model, setup.programs
    spec = trainable_spec(model, setup.subset)
    trainable, frozen = eqx.partition(model, spec)
    losses, grads, heldout_losses = evaluate(setup, trainable, frozen)
    mean = grads.mean(axis=0)
    result = {
        "parameters": parameter_names(trainable),
        "train": [{"start_day": w.start_day, "loss": float(loss),
                   "gradient_norm": float(np.linalg.norm(grad)),
                   "cosine_to_mean_gradient": float(cos)}
                  for w, loss, grad, cos in zip(
                      setup.train, losses, grads, _cosines(grads, mean),
                      strict=True)],
        "heldout": [{"start_day": w.start_day, "loss": float(loss)}
                    for w, loss in zip(setup.heldout, heldout_losses,
                                       strict=True)],
        "mean_gradient_norm": float(np.linalg.norm(mean)),
        "mean_train_loss": float(losses.mean()),
        "mean_heldout_loss": float(heldout_losses.mean()),
    }
    if setup.subset == "gamma":
        derivative = grads[:, 0]
        rows = []
        for step in FD_STEPS:
            fd = (window_values(programs, *eqx.partition(eqx.tree_at(
                      lambda m: m.gamma, model,
                      jnp.asarray(GAMMA_INITIAL + step)), spec),
                      setup.train, gradients=False)[0]
                  - window_values(programs, *eqx.partition(eqx.tree_at(
                      lambda m: m.gamma, model,
                      jnp.asarray(GAMMA_INITIAL - step)), spec),
                      setup.train, gradients=False)[0]) / (2.0 * step)
            rows.append({"step": step, "fd": fd.tolist(),
                         "ad": derivative.tolist(),
                         "relative_error": (np.abs(fd - derivative)
                                            / np.abs(derivative)).tolist(),
                         "mean_fd": float(fd.mean()),
                         "mean_ad": float(derivative.mean())})
        result["gamma_finite_differences"] = rows
    optimizer = create_swg_optimizer(setup.optimizer_settings._replace(
        learning_rate=lr, weight_decay=weight_decay))
    stepped, _ = adam_step(optimizer, optimizer.init(trainable), trainable,
                           frozen, _unravel(trainable, mean))
    new_trainable, new_frozen = eqx.partition(stepped, spec)
    after_train, _ = window_values(
        programs, new_trainable, new_frozen, setup.train, gradients=False)
    after_heldout, _ = window_values(
        programs, new_trainable, new_frozen, setup.heldout, gradients=False)
    result["adam_step"] = {
        "learning_rate": lr, "weight_decay": weight_decay,
        "mean_train_loss_before": float(losses.mean()),
        "mean_train_loss_after": float(after_train.mean()),
        "mean_heldout_loss_before": float(heldout_losses.mean()),
        "mean_heldout_loss_after": float(after_heldout.mean()),
        "train_loss_after_per_start": after_train.tolist(),
        "heldout_loss_after_per_start": after_heldout.tolist()}
    return result, grads


def _unravel(trainable, flat: np.ndarray):
    """The gradient pytree of ``trainable``'s structure from a flat vector."""
    return ravel_pytree(trainable)[1](jnp.asarray(flat))


class Setup(NamedTuple):
    model: ScaledClosure
    subset: str
    programs: tuple
    optimizer_settings: SWGOptimizerSettings
    train: list
    heldout: list


def prepare(args) -> tuple[Setup, dict]:
    """Everything one run needs, from the command-line arguments."""
    config = driver.load_config(args.config)
    if config["schema"] != driver.CONFIG_SCHEMA:
        raise ValueError("--config must be a training configuration")
    device = driver.select_device(
        args.device, "native" if args.device == "cuda" else "libm",
        args.physical_gpu)
    card = driver.build_deck_card(config["deck"], config["surface"])
    geometry = build_closure_geometry(card)
    contract = SWGDatasetContract.from_json(args.contract, card=card)
    if contract.objective != config["objective"]:
        raise ValueError("the contract's objective differs from the configuration's")
    check_start_days(contract, args.window_days, args.train_start_days,
                     args.heldout_start_days)
    model = load_model(config, card, geometry, args.checkpoint)
    setup = Setup(
        model, args.trainable,
        build_programs(card, geometry, days=args.window_days,
                       statistic=args.statistic, precision=args.precision),
        SWGOptimizerSettings(**config["optimiser"]),
        load_windows(contract, card, args.statistic, args.window_days,
                     args.train_start_days),
        load_windows(contract, card, args.statistic, args.window_days,
                     args.heldout_start_days))
    return setup, {"device": device, "config": config, "contract": contract}


def _sha256(path: Path) -> str:
    return driver._sha256(Path(path))


def manifest(args, context: dict) -> dict:
    from legoesm.ocean.fidelity.provenance import git_sha

    arguments = {key: (str(value) if isinstance(value, Path) else value)
                 for key, value in vars(args).items()}
    return {
        "schema": SCHEMA, "legoesm_git_sha": git_sha(allow_dirty=True),
        "command": args.command, "arguments": arguments,
        "start_days": {"train": args.train_start_days,
                       "heldout": args.heldout_start_days},
        "config_sha256": _sha256(args.config),
        "contract_sha256": _sha256(args.contract),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "device": context["device"],
        "gamma_initial": GAMMA_INITIAL, "closure_key_seed": CLOSURE_KEY_SEED,
        "loss": ("sum of the statistic's NRMSE-squared terms, equal weights; "
                 "see the module docstring"),
    }


def write_json(path: Path, payload) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


# ------------------------------------------------------------------ commands

def run_measure(args) -> int:
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists")
    started = time.perf_counter()
    setup, context = prepare(args)
    args.output.mkdir(parents=True)
    write_json(args.output / "manifest.json", manifest(args, context))
    result, grads = measure(setup, lr=args.lr, weight_decay=args.weight_decay)
    result["wall_seconds"] = time.perf_counter() - started
    write_json(args.output / "measure.json", result)
    np.savez(args.output / "gradients.npz", gradients=grads,
             start_days=np.asarray(args.train_start_days))
    print(json.dumps({key: result[key] for key in (
        "mean_train_loss", "mean_heldout_loss", "mean_gradient_norm")}
        | {"adam_step": {k: v for k, v in result["adam_step"].items()
                         if k.startswith("mean_")}}))
    return 0


def _same_run(stored: dict, current: dict) -> bool:
    """Same arguments (the step count may grow) and same input files."""
    def core(manifest):
        return ({k: v for k, v in manifest["arguments"].items() if k != "steps"},
                [manifest[k] for k in ("config_sha256", "contract_sha256",
                                       "checkpoint_sha256")])
    return core(stored) == core(current)


def run_train(args) -> int:
    setup, context = prepare(args)
    spec = trainable_spec(setup.model, setup.subset)
    optimizer = create_swg_optimizer(setup.optimizer_settings._replace(
        learning_rate=args.lr, weight_decay=args.weight_decay))
    trainable, frozen = eqx.partition(setup.model, spec)
    opt_state = optimizer.init(trainable)
    current = manifest(args, context)
    log_path = args.output / "train_log.json"
    if args.output.exists():
        if not _same_run(json.loads((args.output / "manifest.json").read_text()),
                         current):
            raise ValueError(f"{args.output} holds a different run; refusing "
                             "to resume it")
        log = json.loads(log_path.read_text())
        if log["status"] == "nan":
            raise RuntimeError(f"{args.output} stopped on a non-finite value")
    else:
        args.output.mkdir(parents=True)
        log = {"schema": SCHEMA, "status": "running", "steps": []}
        save_checkpoint((setup.model, opt_state),
                        args.output / "step_00000.eqx")
    if len(log["steps"]) > args.steps + 1:
        raise ValueError(f"{args.output} already holds {len(log['steps'])} "
                         f"entries, more than --steps {args.steps} asks for")
    # Entry k holds the losses and the mean gradient at the parameters saved
    # in step_k.eqx; the step to k + 1 is saved before entry k is logged.  A
    # resume recomputes the last logged entry from its checkpoint (the step
    # that follows it was never taken if the run had ended there).
    first = max(len(log["steps"]) - 1, 0)
    del log["steps"][first:]
    log["status"] = "running"
    write_json(args.output / "manifest.json", current)
    write_json(log_path, log)
    model, opt_state = load_checkpoint_or_fail(
        args.output / f"step_{first:05d}.eqx", (setup.model, opt_state),
        what="train step")
    trainable, frozen = eqx.partition(model, spec)
    for step in range(first, args.steps + 1):
        losses, grads, heldout = evaluate(setup, trainable, frozen)
        mean = grads.mean(axis=0)
        entry = {"step": step, "train_loss": float(losses.mean()),
                 "train_loss_per_start": losses.tolist(),
                 "heldout_loss": float(heldout.mean()),
                 "heldout_loss_per_start": heldout.tolist(),
                 "gradient_norm": float(np.linalg.norm(mean))}
        log["steps"].append(entry)
        print(json.dumps({k: entry[k] for k in (
            "step", "train_loss", "heldout_loss", "gradient_norm")}), flush=True)
        if not (np.all(np.isfinite(losses)) and np.all(np.isfinite(heldout))
                and np.all(np.isfinite(grads))):
            log["status"] = "nan"
            write_json(log_path, log)
            return 3
        if step < args.steps:
            model, opt_state = adam_step(
                optimizer, opt_state, trainable, frozen,
                _unravel(trainable, mean))
            save_checkpoint((model, opt_state),
                            args.output / f"step_{step + 1:05d}.eqx")
            trainable, frozen = eqx.partition(model, spec)
        else:
            log["status"] = "complete"
        write_json(log_path, log)
    return 0


def run_free_run(args) -> int:
    """Free-run one saved ``train`` step as a plain network (``free-run``).

    Everything but the step comes from the training run's ``manifest.json``
    (configuration, checkpoint, subset, optimiser settings); the input files
    must still match its digests.  The exported plain network is never
    overwritten; a complete free run is never overwritten and an incomplete
    one is archived (the driver's rules).
    """
    from legoesm.ocean.fidelity.provenance import git_sha

    stored = json.loads((args.run / "manifest.json").read_text())
    if stored["schema"] != SCHEMA or stored["command"] != "train":
        raise ValueError(f"{args.run} is not a {SCHEMA} train run")
    given = stored["arguments"]
    config_path, base = Path(given["config"]), Path(given["checkpoint"])
    for path, key in ((config_path, "config_sha256"),
                      (Path(given["contract"]), "contract_sha256"),
                      (base, "checkpoint_sha256")):
        if _sha256(path) != stored[key]:
            raise ValueError(f"{path} differs from the {key} of {args.run}")
    config = driver.load_config(config_path)
    if config["schema"] != driver.CONFIG_SCHEMA:
        raise ValueError("the run's configuration is not a training configuration")
    free = config["free_run"]
    device = driver.select_device(
        free["device"], free["math_library"], args.physical_gpu,
        reproducible_cuda=True)
    card = driver.build_deck_card(config["deck"], config["surface"])
    geometry = build_closure_geometry(card)
    model = load_model(config, card, geometry, base)
    optimizer = create_swg_optimizer(SWGOptimizerSettings(
        **config["optimiser"])._replace(learning_rate=given["lr"],
                                        weight_decay=given["weight_decay"]))
    spec = trainable_spec(model, given["trainable"])
    model, _ = load_checkpoint_or_fail(
        args.run / f"step_{args.step:05d}.eqx",
        (model, optimizer.init(eqx.partition(model, spec)[0])),
        what="arm-E step")
    network = model.effective_network()
    plain = create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))
    skeleton = (network, plain.init(eqx.filter(network, eqx.is_inexact_array)))
    if args.network_output.exists():
        raise FileExistsError(f"{args.network_output} exists")
    if (args.output / "manifest.json").is_file():
        raise FileExistsError(f"{args.output} already holds a complete free run")
    args.network_output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(skeleton, args.network_output)
    network, _ = load_checkpoint_or_fail(
        args.network_output, skeleton, what="exported network")
    network_spec, _ = driver.network_spec(config, card, geometry)
    closure = free_run_closure({
        "kind": "trained",
        "run_id": (f"{config['experiment']}/seed{args.seed}/arm_e/"
                   f"{args.run.name}/step{args.step:05d}"),
        "experiment": config["experiment"],
        "config_sha256": stored["config_sha256"],
        "dataset_contract_sha256": stored["contract_sha256"],
        "network_seed": args.seed, "window_days": args.window_days,
        "objective": config["objective"], "wall_ring": config["wall_ring"],
        "network_spec": driver._jsonable(network_spec._asdict()),
        "checkpoint_path": str(args.network_output.resolve()),
        "checkpoint_sha256": _sha256(args.network_output)})
    driver._archive_incomplete(args.output)
    completed = driver.free_run(
        network, card, geometry, args.output, days=free["days"],
        closure=closure, commit=git_sha(), device=device,
        start=driver.resolve_start(config, card))
    print(f"FREE_RUN_COMPLETE root={args.output.resolve()} "
          f"manifest_sha256={_sha256(args.output / 'manifest.json')} "
          f"all_finite={completed['numerical_health']['all_finite']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("free-run")
    command.add_argument("--run", type=Path, required=True,
                         help="a ``train`` output directory")
    command.add_argument("--step", type=int, required=True)
    command.add_argument("--seed", type=int, required=True,
                         help="network_seed of the closure block (the starting "
                              "network's seed; the key it seeds meets noise "
                              "scale 0)")
    command.add_argument("--window-days", type=int, required=True,
                         help="window_days of the closure block (the starting "
                              "network's window; the manifest schema admits "
                              "1, 2, 4 and 8)")
    command.add_argument("--physical-gpu", type=int)
    command.add_argument("--network-output", type=Path, required=True,
                         help="the plain network checkpoint to write")
    command.add_argument("--output", type=Path, required=True,
                         help="the free run's directory")
    for name in ("measure", "train"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True,
                             help="training configuration of the checkpoint")
        command.add_argument("--contract", type=Path, required=True,
                             help="its dataset_contract.json")
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), required=True)
        command.add_argument("--physical-gpu", type=int)
        command.add_argument("--window-days", type=int, required=True)
        command.add_argument("--train-start-days", type=int, nargs="+",
                             required=True)
        command.add_argument("--heldout-start-days", type=int, nargs="+",
                             required=True)
        command.add_argument("--trainable", choices=SUBSETS, required=True)
        command.add_argument("--statistic", choices=STATISTICS, required=True)
        command.add_argument("--precision", choices=PRECISIONS, required=True)
        command.add_argument("--lr", type=float, required=True)
        command.add_argument("--weight-decay", type=float, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "train":
            command.add_argument("--steps", type=int, required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return {"measure": run_measure, "train": run_train,
                "free-run": run_free_run}[args.command](args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
