#!/usr/bin/env python3
"""Train the SWG online closure (experiment one) from one explicit config.

``train`` runs one network seed of one configuration file: the round-6
dataset contract over the hourly dual record, the configured window lengths
(of 1, 2, 4 and 8 days) and epochs, batches of ten, Eq. 7, AdamW at a
constant rate through the strict SWG wrapper (re-initialised at every window
length, network carried) and a checkpoint after every epoch.  It resumes
from the last completed epoch of its run directory.  A completed run's
curriculum may be extended by a configuration that only appends window
lengths (same epochs per window and everything else): the finished stages
stay as logged, the new stage starts from the last checkpoint's network
with a fresh optimiser, and the run manifest and training log record the
extension (from which windows to which, from which commit to which).
An optional ``network.initial_checkpoint`` (absolute path of a checkpoint in
this driver's format; no default: absent means the seed's random
initialisation) initialises the network from that file before the first
stage, the optimiser starting fresh; the file's SHA-256 is recorded in the
run manifest and the training log, and a missing, structurally different or
later changed file refuses the run.

``training_precision`` (``"fp64"`` or ``"fp32"``, required) fixes the
arithmetic of every training and validation loss and gradient.  With
``"fp32"`` the float64 master weights are cast to float32 for each call, the
trajectory, loss and gradient are computed in float32, and the gradient is
cast back to float64 before the float64 optimiser update; checkpoints, resume
and the float64 free runs are unchanged.

``gradient_checkpointing`` (``true`` or ``false``, required) selects whether
the segment rematerialises every production step in the backward pass; the
loss and gradient are the same either way, ``false`` trading memory for time.
``dataset.cache`` (``"none"`` or ``"memory"``, required) selects whether
training holds every stored hour its batches can read in host memory
(``SWGDatasetContract.preload_memory``, loaded once through the same
digest-pinned sources) or reads the day files for every batch; batches are
identical.

``free-run`` integrates the coarse card from rest with the closure of one
trained window length (its last checkpoint) on the device and math policy
the configuration states, writing every day's end-of-day snapshot beside its
daily mean in the one free-run schema the held-out scorer admits
(``legoesm.training.swg_online``).  It is a separate process, so a GPU
training run and a configured CPU/libm or CUDA/native free run do not share
an initialized JAX backend.

``profile`` times one optimiser step per window length (batch 10), one
validation batch and, on the free runs' configured device, one free-run day,
and extrapolates the cost of one seed.  ``check-data`` runs the record
through every split and window length before any training.  No command has
a default for any scientific choice: the configuration file states them all
and the run manifest records them.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import functools
import hashlib
import importlib.util
import json
import operator
import os
import re
import resource
import shutil
import sys
import tempfile
import time
import types
import warnings
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ml.checkpoint_io import load_checkpoint_or_fail
from legoesm.ml.training import save_checkpoint
from legoesm.ocean.dynamics.swe_rk3 import SWE_SURFACE_MODES
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_DG50_NAMELIST,
    SWG_DG50_RL_NAMELIST,
    build_swg_card,
    swg_surface_wind,
)
from legoesm.ocean.fidelity.swg_online_decks import (
    SWG_DG50_NU80_NAMELIST,
    require_surface_for_deck,
)
from legoesm.ocean.state import OceanSurfaceForcing
from legoesm.training.swg_online import (
    ADMITTED_WINDOWS_DAYS,
    CUDA_AUTOTUNE_FLAG,
    DAILY_MEAN_FIELDS,
    DAILY_MEAN_LABEL,
    DUAL_SCHEMA,
    FREE_RUN_EXECUTIONS,
    FREE_RUN_MANIFEST_SCHEMA,
    HOURS_PER_DAY,
    OBJECTIVE_OPERATORS,
    SNAPSHOT_FIELDS,
    SNAPSHOT_LABEL,
    START_REBALANCE_MAX_ITERATIONS,
    START_REBALANCE_TOLERANCE,
    TEMPLATE_BATCH_SIZE,
    WALL_RING_OPTIONS,
    SWGDatasetContract,
    SWGNetworkSpec,
    SWGOptimizerSettings,
    SWGScalarPotentialCNN,
    WindowBatch,
    assert_network_gradients_live,
    build_closure_geometry,
    build_segment_fn,
    coarse_state_from_native,
    create_swg_optimizer,
    deterministic_trajectory_loss,
    eq6_scale_report,
    free_run_closure,
    initial_closure_acceleration_ratio,
    maddison_eq6_network_scales,
    maddison_half_width_m,
    network_parameter_count,
    record_start_state,
    require_rigid_lid_zero_ssh,
    rigid_lid_start_state,
    start_state_from_native,
    steps_per_target_interval,
    swg_online_step,
    window_steps,
)

CONFIG_SCHEMA = "legoesm-swg-online-training-v3"
CONTROL_CONFIG_SCHEMA = "legoesm-swg-online-control-v1"
RUN_SCHEMA = "legoesm-swg-online-training-run-v1"
LOG_SCHEMA = "legoesm-swg-online-training-log-v2"
PROFILE_SCHEMA = "legoesm-swg-online-training-profile-v1"
DATASET_SCHEMA = "legoesm-swg-online-dataset-v1"
SCALE_RULE = "maddison_eq6_shallow_water_grid_analogue"
DECKS = {"dg50": SWG_DG50_NAMELIST, "dg50-nu80": SWG_DG50_NU80_NAMELIST,
         "dg50-rl": SWG_DG50_RL_NAMELIST}
TRAINING_DECKS = ("dg50", "dg50-rl")
CONTROL_DECKS = {"nu10": "dg50", "nu80": "dg50-nu80"}
FREE_RUN_SAMPLING = "end_of_day_snapshot_and_daily_mean"
TRAINING_PRECISIONS = ("fp64", "fp32")
DATASET_CACHES = ("none", "memory")
SECONDS_PER_DAY = 86_400
_OPERATOR_FIELDS = {"symmetric": "box", "template": "filtered"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    return json.loads(json.dumps(value))


@functools.cache
def _double_gyre_driver():
    """The SWG double-gyre driver, loaded by path (scripts are not packages)."""
    path = Path(__file__).with_name("run_swg_double_gyre.py")
    spec = importlib.util.spec_from_file_location("swg_double_gyre_driver", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("cannot load the SWG double-gyre driver")
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    _double_gyre_driver()._write_json_atomic(path, payload)


# --------------------------------------------------------------- configuration

def _require_keys(mapping, expected, where: str) -> None:
    if not isinstance(mapping, dict) or set(mapping) != set(expected):
        found = sorted(mapping) if isinstance(mapping, dict) else type(mapping)
        raise ValueError(
            f"{where} must hold exactly the keys {sorted(expected)}, "
            f"found {found}")


def _validate_surface(surface) -> None:
    if surface not in SWE_SURFACE_MODES:
        raise ValueError(f"surface must be one of {SWE_SURFACE_MODES}")


def _require_int(value, where: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{where} must be an integer >= {minimum}")
    return value


def _validate_free_run(free_run: dict) -> None:
    _require_keys(free_run, ("days", "start", "sampling", "device",
                             "math_library"), "free_run")
    _require_int(free_run["days"], "free_run days", minimum=1)
    if free_run["sampling"] != FREE_RUN_SAMPLING:
        raise ValueError(f"free_run sampling must be {FREE_RUN_SAMPLING!r}")
    start = free_run["start"]
    if isinstance(start, dict):
        _require_keys(start, ("kind", "record", "day", "objective"),
                      "free_run start")
        if start["kind"] != "record_state":
            raise ValueError(
                "free_run start must be 'rest' or a record_state start")
        _require_keys(start["record"], ("root", "chunk_label"),
                      "free_run start record")
        _require_int(start["day"], "free_run start day", minimum=0)
        if start["objective"] not in OBJECTIVE_OPERATORS:
            raise ValueError("free_run start objective must be one of "
                             f"{OBJECTIVE_OPERATORS}")
    elif start != "rest":
        raise ValueError(
            "free_run start must be 'rest' or a record_state start")
    execution = (free_run["device"], free_run["math_library"])
    if execution not in FREE_RUN_EXECUTIONS:
        raise ValueError(
            "free-run device and math_library must be either "
            "cpu+libm or cuda+native")


def load_config(path: Path) -> dict:
    """Read one training configuration; every key is required, none extra."""
    config = json.loads(Path(path).read_text())
    if config.get("schema") == CONTROL_CONFIG_SCHEMA:
        _require_keys(config, (
            "schema", "experiment", "description", "control", "deck",
            "surface", "closure", "free_run"), "control configuration")
        if not isinstance(config["experiment"], str) \
                or re.fullmatch(r"[a-z0-9_]+", config["experiment"]) is None:
            raise ValueError("experiment must be a lowercase [a-z0-9_] name")
        if not isinstance(config["description"], str) or not config["description"]:
            raise ValueError("description must be a non-empty string")
        if config["control"] not in CONTROL_DECKS:
            raise ValueError(f"control must be one of {sorted(CONTROL_DECKS)}")
        if config["deck"] != CONTROL_DECKS[config["control"]]:
            raise ValueError(
                f"control {config['control']} requires deck "
                f"{CONTROL_DECKS[config['control']]!r}")
        _validate_surface(config["surface"])
        if config["closure"] != "none":
            raise ValueError("control free runs require closure='none'")
        _validate_free_run(config["free_run"])
        return config

    _require_keys(config, (
        "schema", "experiment", "description", "deck", "surface", "objective",
        "wall_ring", "training_precision", "gradient_checkpointing",
        "target_interval_hours", "network", "seeds", "dataset",
        "curriculum", "optimiser", "free_run"), "configuration")
    if config["schema"] != CONFIG_SCHEMA:
        raise ValueError(f"configuration schema must be {CONFIG_SCHEMA!r}")
    if not isinstance(config["experiment"], str) \
            or re.fullmatch(r"[a-z0-9_]+", config["experiment"]) is None:
        raise ValueError("experiment must be a lowercase [a-z0-9_] name")
    if not isinstance(config["description"], str) or not config["description"]:
        raise ValueError("description must be a non-empty string")
    if config["deck"] not in TRAINING_DECKS:
        raise ValueError(f"training deck must be one of {TRAINING_DECKS}")
    _validate_surface(config["surface"])
    require_surface_for_deck(config["deck"], config["surface"])
    if config["objective"] not in OBJECTIVE_OPERATORS:
        raise ValueError(f"objective must be one of {OBJECTIVE_OPERATORS}")
    if config["wall_ring"] not in WALL_RING_OPTIONS:
        raise ValueError(f"wall_ring must be one of {WALL_RING_OPTIONS}")
    if config["training_precision"] not in TRAINING_PRECISIONS:
        raise ValueError(
            f"training_precision must be one of {TRAINING_PRECISIONS}")
    if type(config["gradient_checkpointing"]) is not bool:
        raise ValueError("gradient_checkpointing must be true or false")
    # A valid interval whose whole number of deck steps fits (refused here,
    # not mid-run).
    steps_per_target_interval(
        DECKS[config["deck"]].rn_Dt_s, config["target_interval_hours"])

    network = config["network"]
    _require_keys({key: value for key, value in network.items()
                   if key != "initial_checkpoint"},
                  ("hidden_channels", "hidden_layers", "kernel_size",
                   "wall_margin_rings", "scales"), "network")
    if "initial_checkpoint" in network and (
            not isinstance(network["initial_checkpoint"], str)
            or not Path(network["initial_checkpoint"]).is_absolute()):
        raise ValueError("network initial_checkpoint must be an absolute path")
    _require_int(network["hidden_channels"], "hidden_channels", minimum=1)
    _require_int(network["hidden_layers"], "hidden_layers", minimum=1)
    if _require_int(network["kernel_size"], "kernel_size", minimum=1) % 2 != 1:
        raise ValueError("kernel_size must be odd")
    _require_int(network["wall_margin_rings"], "wall_margin_rings", minimum=0)
    if network["scales"] != SCALE_RULE:
        raise ValueError(f"network scales must be {SCALE_RULE!r}")

    # One or more: a pilot trains one seed; the held-out gate scores only
    # three or more distinct runs (the scorer refuses fewer).
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must list one or more distinct seeds")
    for seed in seeds:
        _require_int(seed, "seed", minimum=0)
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must list one or more distinct seeds")

    dataset = config["dataset"]
    _require_keys(dataset, ("records", "splits_days", "batch_size", "cache"),
                  "dataset")
    if dataset["cache"] not in DATASET_CACHES:
        raise ValueError(f"dataset cache must be one of {DATASET_CACHES}")
    if dataset["batch_size"] != TEMPLATE_BATCH_SIZE:
        raise ValueError(f"batch_size must be {TEMPLATE_BATCH_SIZE}")
    records = dataset["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("dataset records must be a non-empty list")
    for record in records:
        _require_keys(record, ("root", "chunk_label"), "dataset record")
        if not isinstance(record["root"], str) \
                or not Path(record["root"]).is_absolute():
            raise ValueError("dataset record root must be an absolute path")
        # The test-year chunks on disk are dual_test_dayAAAA_to_dayBBBB.
        if not isinstance(record["chunk_label"], str) \
                or re.fullmatch(r"[a-z0-9_]+", record["chunk_label"]) is None:
            raise ValueError("chunk_label must be a lowercase [a-z0-9_] word")
    if len({record["root"] for record in records}) != len(records):
        raise ValueError("dataset record roots are duplicated")
    splits = dataset["splits_days"]
    _require_keys(splits, ("train", "validation", "test"), "splits_days")
    for name, bounds in splits.items():
        if not isinstance(bounds, list) or len(bounds) != 2 \
                or any(type(day) is not int for day in bounds) \
                or not 0 <= bounds[0] < bounds[1]:
            raise ValueError(f"split {name} must be [first_day, last_day]")

    curriculum = config["curriculum"]
    _require_keys(curriculum, ("window_days", "epochs_per_window",
                               "network_between_windows",
                               "optimiser_between_windows"), "curriculum")
    windows = curriculum["window_days"]
    if not isinstance(windows, list) or not windows \
            or any(window not in ADMITTED_WINDOWS_DAYS for window in windows) \
            or windows != sorted(set(windows)):
        raise ValueError(
            f"window_days must increase through {ADMITTED_WINDOWS_DAYS}")
    _require_int(curriculum["epochs_per_window"], "epochs_per_window", minimum=1)
    if curriculum["network_between_windows"] != "carried":
        raise ValueError("network_between_windows must be 'carried'")
    if curriculum["optimiser_between_windows"] != "reinitialised":
        raise ValueError("optimiser_between_windows must be 'reinitialised'")

    _require_keys(config["optimiser"], SWGOptimizerSettings._fields, "optimiser")
    create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))

    _validate_free_run(config["free_run"])
    start = config["free_run"]["start"]
    if isinstance(start, dict) and start["objective"] != config["objective"]:
        raise ValueError("a trained free run's record start must use the "
                         "training objective's operator")
    return config


def resolve_start(config: dict, card):
    """The configured free-run start: ``(state or None, manifest block)``.

    ``"rest"`` is the card's resting state (``None`` here; ``free_run`` seeds
    it).  A ``record_state`` start is the stored coarse state of the named
    record at the end of ``day`` (absolute hour ``24 * day``), sea level
    rebalanced exactly as every training window start
    (``record_start_state``); the manifest block records the serving chunk,
    its digest and the rebalancing report.
    """
    start = config["free_run"]["start"]
    if start == "rest":
        return None, "rest"
    chunks = [str(path) for _, _, path in complete_chunks(start["record"])]
    if not chunks:
        raise ValueError(f"free_run start record {start['record']['root']} "
                         "holds no complete chunk")
    (ssh, u, v), provenance = record_start_state(
        card, chunks, objective=start["objective"],
        absolute_hour=HOURS_PER_DAY * start["day"])
    state = start_state_from_native(card, ssh, u, v)
    return state, _jsonable(start | provenance)


# ---------------------------------------------------------------- device/card

def select_device(device: str, math_library: str, physical_gpu: int | None, *,
                  reproducible_cuda: bool = False, environ=None) -> dict:
    """Check the explicit device selection and set the fp64 policy.

    A GPU run needs ``JAX_PLATFORMS=cuda`` (JAX 0.10 refuses ``gpu``) and
    ``CUDA_VISIBLE_DEVICES`` equal to the named physical GPU; a CPU run
    needs ``JAX_PLATFORMS=cpu``.  Nothing is inferred from what is visible.
    """
    environment = os.environ if environ is None else environ
    if (device, math_library) not in FREE_RUN_EXECUTIONS:
        raise ValueError(
            "device and math_library must be cpu+libm or cuda+native")
    if environment.get("JAX_PLATFORMS") != device:
        raise RuntimeError(
            f"JAX_PLATFORMS must be exactly {device!r}, found "
            f"{environment.get('JAX_PLATFORMS')!r}")
    if device == "cuda":
        if type(physical_gpu) is not int or physical_gpu < 0:
            raise ValueError("a cuda run requires --physical-gpu")
        if environment.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES differs from --physical-gpu: "
                f"{environment.get('CUDA_VISIBLE_DEVICES')!r} != "
                f"{physical_gpu}")
        if reproducible_cuda:
            flags = environment.get("XLA_FLAGS", "").split()
            autotune = [flag for flag in flags
                        if flag.startswith("--xla_gpu_autotune_level=")]
            if autotune and autotune != [CUDA_AUTOTUNE_FLAG]:
                raise RuntimeError(
                    "XLA_FLAGS sets a conflicting xla_gpu_autotune_level; "
                    f"free runs require {CUDA_AUTOTUNE_FLAG}")
            if not autotune:
                flags.append(CUDA_AUTOTUNE_FLAG)
                environment["XLA_FLAGS"] = " ".join(flags)
    elif physical_gpu is not None:
        raise ValueError("--physical-gpu is only valid with --device cuda")
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("JAX x64 is disabled (set JAX_ENABLE_X64=1)")
    backend = jax.default_backend()
    if backend != ("gpu" if device == "cuda" else "cpu"):
        raise RuntimeError(f"device {device} selected backend {backend!r}")
    set_policy(PrecisionPolicy.fp64(transcendentals=math_library))
    devices = jax.devices()
    gpu_name = devices[0].device_kind if device == "cuda" else None
    return {
        "device": device, "backend": backend, "precision_policy": "fp64",
        "math_library": math_library, "physical_gpu": physical_gpu,
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        "jax_platforms": environment.get("JAX_PLATFORMS"),
        "xla_flags": environment.get("XLA_FLAGS", ""),
        "jax_version": jax.__version__, "gpu_name": gpu_name,
        "xla_python_client_mem_fraction": environment.get(
            "XLA_PYTHON_CLIENT_MEM_FRACTION"),
    }


def build_deck_card(deck: str, surface: str):
    """The coarse card of one named deck (after ``select_device``)."""
    if deck not in DECKS:
        raise ValueError(f"deck must be one of {sorted(DECKS)}")
    _validate_surface(surface)
    require_surface_for_deck(deck, surface)
    return build_swg_card(DECKS[deck], surface=surface)


def steps_per_day(card) -> int:
    steps = SECONDS_PER_DAY / card.dt_s
    if not float(steps).is_integer():
        raise ValueError("the coarse timestep must divide one day")
    return int(steps)


def free_run_namelist(card, days: int):
    """The card's deck resolved for a ``days``-long run with daily output,
    as the double-gyre driver resolves its decks (``nn_write`` = one day)."""
    per_day = steps_per_day(card)
    return card.resolved_namelist._replace(
        nn_itend=days * per_day, nn_write=per_day, nn_stock=per_day)


def network_spec(config: dict, card, geometry) -> tuple:
    scales = maddison_eq6_network_scales(card, geometry)
    network = config["network"]
    return SWGNetworkSpec(
        hidden_channels=network["hidden_channels"],
        hidden_layers=network["hidden_layers"],
        kernel_size=network["kernel_size"],
        ssh_scale_m=scales.ssh_scale_m,
        velocity_scale_m_s=scales.velocity_scale_m_s,
        potential_scale_m3_s2=scales.potential_scale_m3_s2,
        wall_margin_rings=network["wall_margin_rings"]), scales


# --------------------------------------------------------------------- dataset

def complete_chunks(record: dict) -> list[tuple[int, int, Path]]:
    """Complete chunks ``<label>_dayAAAA_to_dayBBBB`` of one record root.

    A chunk still being written (no ``manifest.json``, or a status other
    than complete) is skipped; a complete chunk whose manifest names a
    different restart day or length than its label raises.
    """
    pattern = re.compile(
        rf"{re.escape(record['chunk_label'])}_day(\d{{4}})_to_day(\d{{4}})")
    chunks = []
    if not Path(record["root"]).is_dir():
        raise ValueError(f"dataset record root does not exist: {record['root']}")
    for path in sorted(Path(record["root"]).iterdir()):
        match = pattern.fullmatch(path.name)
        manifest_path = path / "manifest.json"
        if match is None or not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            continue
        first, last = int(match[1]), int(match[2])
        if (manifest.get("restart") or {}).get("source_day") != first \
                or manifest.get("days") != last - first:
            raise ValueError(
                f"{path}: label differs from its manifest's restart day/length")
        chunks.append((first, last, path))
    return chunks


def write_dataset_contract(config: dict, path: Path) -> dict:
    """Pin every complete chunk that touches a split in a round-6 contract."""
    splits = config["dataset"]["splits_days"]
    sources = []
    for record in config["dataset"]["records"]:
        for first, last, chunk in complete_chunks(record):
            if any(first <= end and last >= start
                   for start, end in splits.values()):
                sources.append({
                    "root": str(chunk.resolve()),
                    "manifest_sha256": _sha256(chunk / "manifest.json")})
    payload = {
        "schema": DATASET_SCHEMA, "objective": config["objective"],
        "batch_size": config["dataset"]["batch_size"],
        "window_days": list(ADMITTED_WINDOWS_DAYS),
        "splits": {name: [HOURS_PER_DAY * start, HOURS_PER_DAY * end]
                   for name, (start, end) in splits.items()},
        "sources": sources,
    }
    _write_json(path, payload)
    return payload


def check_data(config: dict, *, card) -> dict:
    """Run the record through every split and window length before training.

    Pins and validates the contract (in a temporary directory; a training
    run pins its own) and builds the first batch of each split at each
    curriculum window length: training at epoch 0 with the first seed,
    held-out splits on their fixed lattice.  Loading checks every day file's
    digest, record count and finiteness, and every start is rebalanced; any
    defect raises.
    """
    with tempfile.TemporaryDirectory(prefix="swgonl-check-") as folder:
        path = Path(folder) / "dataset_contract.json"
        payload = write_dataset_contract(config, path)
        contract = SWGDatasetContract.from_json(path, card=card)
        rows = []
        for window_days in config["curriculum"]["window_days"]:
            for split in ("train", "validation", "test"):
                sampler = ({"epoch": 0, "seed": config["seeds"][0]}
                           if split == "train" else {})
                batch = next(contract.batches(
                    split=split, window_days=window_days,
                    target_interval_hours=config["target_interval_hours"],
                    **sampler))
                rows.append({
                    "split": split, "window_days": window_days,
                    "first_batch_start_hours": [
                        int(hour) for hour in batch.start_absolute_hour]})
        return {"contract_sha256": _sha256(path),
                "source_count": len(payload["sources"]),
                "splits_hours": payload["splits"], "batches": rows}


# ------------------------------------------------------------ batch programs

@contextlib.contextmanager
def _single_precision_trace():
    """Trace and run with x64 off: every float64 the card, the segment and
    the network request becomes float32 (the card is built in float64)."""
    with jax.enable_x64(False), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Explicitly requested dtype.*")
        yield


def _single(tree):
    return jax.tree.map(
        lambda leaf: leaf.astype(jnp.float32)
        if eqx.is_inexact_array(leaf) else leaf, tree)


def _in_single_precision(function):
    """``function`` on float32 copies of the weights and the batch, traced
    with x64 off; its outputs (loss, gradient) come back as float64."""
    def run(network, initial, target, key):
        with _single_precision_trace():
            result = function(_single(network), _single(initial),
                              _single(target), key)
        return jax.tree.map(lambda leaf: leaf.astype(jnp.float64), result)

    return run


def build_batch_functions(card, geometry, *, window_days: int,
                          target_interval_hours: int,
                          objective: str, wall_ring: str,
                          training_precision: str, checkpoint: bool):
    """Jitted (value-and-grad, value) of the Eq. 7 loss of a window batch.

    Every window starts from its (rebalanced) stored state and runs the
    certified segment with the closure at every RK3 stage, zero stage noise
    and zero noise scale; the batch loss is ``deterministic_trajectory_loss``
    over all windows and recorded endpoints, one every
    ``target_interval_hours`` hours (Eq. 7's mean, equal weights).

    ``training_precision`` ``"fp32"`` makes both functions take the float64
    master weights and batch, compute the trajectory, loss and gradient in
    float32 (x64 off for the build, trace and call only) and return them in
    float64; ``"fp64"`` computes in float64 throughout.
    ``checkpoint`` is passed to ``build_segment_fn``: ``True`` rematerialises
    every step in the backward pass, ``False`` keeps
    the activations (same loss and gradient, more memory, less time).
    """
    if training_precision not in TRAINING_PRECISIONS:
        raise ValueError(
            f"training_precision must be one of {TRAINING_PRECISIONS}")
    if type(checkpoint) is not bool:
        raise ValueError("checkpoint must be true or false")
    single = training_precision == "fp32"
    with _single_precision_trace() if single else contextlib.nullcontext():
        if single:
            card = card._replace(recipe=card.recipe._replace(
                grid=_single(card.recipe.grid),
                land_mask=_single(card.recipe.land_mask),
                initial_state=_single(card.recipe.initial_state)))
            geometry = _single(geometry)
        raw, _ = build_segment_fn(
            card, geometry, window_days=window_days,
            target_interval_hours=target_interval_hours, checkpoint=checkpoint)
    n_steps = window_steps(card, window_days, target_interval_hours)
    template = card.recipe.initial_state
    namelist = card.resolved_namelist
    shape = tuple(geometry.tmask.shape)
    noise_dtype = jnp.float32 if single else jnp.float64

    def batch_loss(network, initial, target, key):
        noise = jnp.zeros((n_steps, 3) + shape, dtype=noise_dtype)

        def rollout(ssh, u, v):
            state = rigid_lid_start_state(
                card, coarse_state_from_native(template, ssh, u, v))
            return raw(network, state, noise, key, jnp.asarray(0.0))[1]

        predicted = jax.vmap(rollout)(*initial)
        return deterministic_trajectory_loss(
            predicted, target, geometry, objective=objective,
            beta_m1_s1=namelist.rn_beta_m_s,
            basin_length_m=namelist.rn_domsiz_m, wall_ring=wall_ring)

    value_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(batch_loss))
    loss_fn = eqx.filter_jit(batch_loss)
    if single:
        value_and_grad = _in_single_precision(value_and_grad)
        loss_fn = _in_single_precision(loss_fn)

    def zero_lid_ssh(function):
        # Under the lid the traced start cannot refuse stored sea level;
        # the concrete batch is refused here (no-op on an explicit card).
        def run(network, initial, target, key):
            require_rigid_lid_zero_ssh(card, initial[0])
            return function(network, initial, target, key)
        return run

    return zero_lid_ssh(value_and_grad), zero_lid_ssh(loss_fn)


def _parts(batch, windows_per_call: int):
    for first in range(0, TEMPLATE_BATCH_SIZE, windows_per_call):
        part = slice(first, first + windows_per_call)
        yield (tuple(jnp.asarray(value[part]) for value in batch.initial),
               tuple(jnp.asarray(value[part]) for value in batch.target))


def batch_value_and_grad(value_and_grad, network, batch, key, windows_per_call):
    """Batch loss and gradient from equal parts of the ten windows.

    Each part's loss is the Eq. 7 mean over its windows, so the mean of the
    parts is the batch's Eq. 7 loss exactly (only the summation order
    depends on ``windows_per_call``).
    """
    results = [value_and_grad(network, initial, target, key)
               for initial, target in _parts(batch, windows_per_call)]
    count = len(results)
    loss = functools.reduce(operator.add, [value for value, _ in results])
    grads = jax.tree.map(lambda *parts: functools.reduce(operator.add, parts),
                         *[grad for _, grad in results])
    return loss / count, jax.tree.map(lambda leaf: leaf / count, grads)


def batch_loss_value(loss_fn, network, batch, key, windows_per_call) -> float:
    values = [loss_fn(network, initial, target, key)
              for initial, target in _parts(batch, windows_per_call)]
    return float(functools.reduce(operator.add, values) / len(values))


def _require_finite(loss, grads, where: str) -> None:
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
    if not np.isfinite(float(loss)) \
            or not all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves):
        raise RuntimeError(f"non-finite loss or gradient at {where}")


def batch_start_states(card, batch) -> tuple:
    return tuple(start_state_from_native(
        card, *(value[index] for value in batch.initial))
        for index in range(TEMPLATE_BATCH_SIZE))


# ------------------------------------------------------------------- free run

def free_run_day(network, card, geometry, *, key):
    """One compiled model day of a free run, and the steady forcing.

    The double-gyre driver's daily-mean stepper around ``swg_online_step``
    (closure at every RK3 stage, zero stage noise, noise scale 0;
    ``network=None`` steps without a closure): ``state -> (end-of-day state,
    (ssh, u, v) arithmetic means of every post-step state of the day)``.
    """
    tau_u, tau_v = swg_surface_wind(card.resolved_namelist)
    forcing = OceanSurfaceForcing(
        tau_i_native=jnp.asarray(tau_u, dtype=jnp.float64),
        tau_j_native=jnp.asarray(tau_v, dtype=jnp.float64))
    stage_noise = jnp.zeros((3,) + tuple(geometry.tmask.shape), jnp.float64)

    def step(state, dt_s, surface_forcing):
        if dt_s != card.dt_s:
            raise ValueError("a free-run step must use the card's timestep")
        return swg_online_step(
            network, state, stage_noise, key, jnp.asarray(0.0), card=card,
            geometry=geometry, forcing=surface_forcing)

    return forcing, _double_gyre_driver().build_scanned_day_mean_stepper(
        types.SimpleNamespace(step=step), card.dt_s, forcing,
        steps_per_day(card))


def free_run(network, card, geometry, output: Path, *, days: int,
             closure: dict, commit: str, device: dict, start) -> dict:
    """Integrate the coarse card with the closure, dual day files.

    ``start`` is ``resolve_start``'s pair: the initial state (``None`` for
    the card's rest state) and the manifest's ``start`` block.

    One compiled day (``free_run_day``) per model day.  Every day file holds
    the end-of-day snapshot and that day's mean of every post-step state
    under the dual per-file labels the held-out scorer admits; the manifest
    is the one free-run schema, its closure block validated (checkpoint file
    included) by ``free_run_closure`` before anything is written.
    """
    driver = _double_gyre_driver()
    closure = free_run_closure(closure)
    per_day = steps_per_day(card)
    namelist = free_run_namelist(card, days)
    key = (jax.random.PRNGKey(closure["network_seed"])
           if closure["kind"] == "trained" else None)
    forcing, one_day = free_run_day(network, card, geometry, key=key)
    initial, start_block = start
    state = driver.seed_steady_forcing_carry(
        card.recipe.initial_state if initial is None else initial, forcing)
    output.mkdir(parents=True)
    manifest = {
        "schema": FREE_RUN_MANIFEST_SCHEMA, "legoesm_git_sha": commit,
        "device": device["device"], "backend": device["backend"],
        "precision_policy": "fp64", "math_library": device["math_library"],
        "xla_flags": device["xla_flags"], "jax_version": device["jax_version"],
        "gpu_name": device["gpu_name"], "days": days,
        "steps_per_day": per_day, "start": start_block,
        "integration": ("one jitted scan per model day of swg_online_step "
                        "(certified SWE RK3 step, closure potential at every "
                        "stage, zero stage noise, noise scale 0) in the "
                        "double-gyre driver's daily-mean stepper"),
        "sampling": FREE_RUN_SAMPLING,
        "fields": {"ssh_m": "end-of-day snapshot; T points",
                   "u_m_s": "end-of-day snapshot; native NEMO U record",
                   "v_m_s": "end-of-day snapshot; native NEMO V record",
                   "ssh_daily_mean_m": "daily mean of post-step T-point SSH",
                   "u_daily_mean_m_s": "daily mean of post-step native U velocity",
                   "v_daily_mean_m_s": "daily mean of post-step native V velocity"},
        "namelist": namelist._asdict(), "closure": closure,
        "numerical_health": None,
    }
    pending = output / "manifest.pending.json"
    hashes: dict[str, str] = {}
    _write_json(pending, manifest | {"status": "pending", "completed_days": 0,
                                     "daily_sha256": hashes})
    health = None
    started = time.perf_counter()
    for day in range(1, days + 1):
        state, means = one_day(state)
        jax.block_until_ready(state.eta.data)
        if day == 1:
            first_day_s = time.perf_counter() - started
        daily = driver.numerical_health(state, namelist)
        health = daily if health is None else {
            "all_finite": health["all_finite"] and daily["all_finite"],
            "minimum_total_layer_thickness_m": min(
                health["minimum_total_layer_thickness_m"],
                daily["minimum_total_layer_thickness_m"]),
            "maximum_u_courant": max(health["maximum_u_courant"],
                                     daily["maximum_u_courant"]),
            "maximum_v_courant": max(health["maximum_v_courant"],
                                     daily["maximum_v_courant"])}
        manifest["numerical_health"] = health
        name = f"day{day:04d}.npz"
        temporary = output / f".{name}.tmp"
        snapshot = (state.eta.data, state.u.data[:, 1:, 0],
                    state.v.data[1:, :, 0])
        with temporary.open("wb") as stream:
            np.savez(stream, schema=np.str_(DUAL_SCHEMA),
                     legoesm_git_sha=np.str_(commit), day=np.int64(day),
                     absolute_day=np.int64(day),
                     step=np.int64(day * per_day), dt_s=np.float64(card.dt_s),
                     snapshot_sampling=np.str_(SNAPSHOT_LABEL),
                     daily_mean_sampling=np.str_(DAILY_MEAN_LABEL),
                     daily_mean_sample_count=np.int64(per_day),
                     **{key: np.asarray(value, dtype=np.float64) for key, value
                        in zip(SNAPSHOT_FIELDS + DAILY_MEAN_FIELDS,
                               snapshot + tuple(means)[:3], strict=True)})
        temporary.replace(output / name)
        hashes[name] = _sha256(output / name)
        if day % driver.METADATA_CHECKPOINT_DAYS == 0 or day == days:
            driver._write_hashes_atomic(output / "daily.sha256", hashes)
            _write_json(pending, manifest | {
                "status": "pending", "completed_days": day,
                "daily_sha256": hashes})
    completed = manifest | {
        "status": "complete", "completed_days": days, "daily_sha256": hashes,
        "execution_seconds": {"first_day_with_compile": first_day_s,
                              "all_days": time.perf_counter() - started}}
    _write_json(pending, completed)
    pending.replace(output / "manifest.json")
    return completed


def _archive_incomplete(path: Path) -> None:
    if not path.exists():
        return
    suffix = 0
    archive = path.with_name(path.name + ".interrupted")
    while archive.exists():
        suffix += 1
        archive = path.with_name(f"{path.name}.interrupted.{suffix}")
    path.rename(archive)


# ---------------------------------------------------------------------- train

def _identity(config: dict, config_path: Path, card, geometry, spec, scales,
              contract_sha256: str, contract: dict, *, seed: int,
              device: dict, windows_per_call: int, commit: str) -> dict:
    namelist = card.resolved_namelist
    half_width = maddison_half_width_m(namelist.rn_domsiz_m, geometry)
    per_day = steps_per_day(card)
    interval = config["target_interval_hours"]
    pretrained = config["network"].get("initial_checkpoint")
    network_init = "SWGScalarPotentialCNN(spec, key=PRNGKey(seed))"
    initial_checkpoint = {}
    if pretrained is not None:
        # A missing file raises here (FileNotFoundError), before any run
        # directory exists.
        initial_checkpoint = {"initial_checkpoint": {
            "path": pretrained, "sha256": _sha256(Path(pretrained))}}
        network_init += ("; weights then loaded from "
                         "network.initial_checkpoint (optimiser fresh)")
    return _jsonable({
        "schema": RUN_SCHEMA,
        "run_id": f"{config['experiment']}/seed{seed}",
        "config_sha256": _sha256(config_path),
        "config": config,
        "seed": {
            "value": seed,
            "network_init": network_init,
            "window_sampler": "SWGDatasetContract.batches(seed=seed, "
                              "epoch=epoch-1): offset per (seed, window, "
                              "epoch), shuffled",
            "rollout_key": "PRNGKey(seed); inert: zero stage noise, scale 0"},
        "legoesm_git_sha": commit,
        "versions": {"jax": jax.__version__, "equinox": eqx.__version__,
                     "optax": optax.__version__, "numpy": np.__version__},
        "device": {key: device[key] for key in (
            "device", "backend", "precision_policy", "math_library",
            "xla_flags", "jax_version", "gpu_name")},
        "execution": {
            "windows_per_call": windows_per_call,
            "segment": "build_segment_fn(checkpoint=True): rematerialise "
                       "every production step"
            if config["gradient_checkpointing"] else
            "build_segment_fn(checkpoint=False): keep every production "
            "step's activations"},
        "card": {"namelist": namelist._asdict(), "dt_s": card.dt_s,
                 "steps_per_day": per_day,
                 "grid_shape": list(geometry.tmask.shape)},
        "network": {
            "spec": spec._asdict(),
            "parameter_count": network_parameter_count(
                SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(seed))),
            "scale_rule": SCALE_RULE, "scales": scales._asdict(),
            "input_channels": ["ssh/ssh_scale (+ zero noise)",
                               "T-centred u/velocity_scale",
                               "T-centred v/velocity_scale", "wet mask"],
            "output": "potential_scale x boundary-zero F-point potential",
            **initial_checkpoint},
        "optimiser": {
            "settings": config["optimiser"],
            "constructor": "legoesm.training.swg_online.create_swg_optimizer",
            "between_windows": "reinitialised"},
        "loss": {
            "definition": "Maddison (2026) Eq. 7, deterministic_trajectory_loss",
            "objective": config["objective"],
            "wall_ring": config["wall_ring"],
            "beta_m1_s1": namelist.rn_beta_m_s,
            "basin_length_m": namelist.rn_domsiz_m,
            "half_width_m": half_width,
            "prefactor": 1.0e4 / (namelist.rn_beta_m_s**2 * half_width**2)
            / (4.0 * half_width**2),
            "reduction": "mean over windows and "
                         + ("hourly" if interval == 1
                            else f"{interval}-hourly") + " right endpoints",
            "training_loss_logged": "mean over the epoch's batches, each at "
                                    "its pre-update parameters",
            "validation_loss_logged": "mean over every validation batch of "
                                      "the window length, end of epoch"},
        "dataset": {
            "contract_sha256": contract_sha256,
            "splits_hours": contract["splits"],
            "source_count": len(contract["sources"]),
            "operator": _OPERATOR_FIELDS[config["objective"]],
            "start_state": {
                "rule": "stored state at the start hour, sea level "
                        "rebalanced (build_start_rebalancer)",
                "tolerance": START_REBALANCE_TOLERANCE,
                "max_iterations": START_REBALANCE_MAX_ITERATIONS},
            "target": ("stored right endpoints 1..24w hours later, same "
                       "operator" if interval == 1 else
                       f"every {interval}th stored hour, {interval}..24w "
                       "hours after the window start, same operator")},
        "curriculum": config["curriculum"],
        "gates": {
            "initial_closure_acceleration": "initial_closure_acceleration_"
                                            "ratio on the ten starts of the "
                                            "first training batch, bar 0.01",
            "no_inert": "assert_network_gradients_live on the first "
                        "optimiser step's gradient",
            "finite": "loss and every gradient element at every step"},
        "free_run": config["free_run"] | {"namelist": free_run_namelist(
            card, config["free_run"]["days"])._asdict()},
    })


def _checkpoint_name(window_days: int, epoch: int) -> str:
    return f"checkpoints/w{window_days:02d}_e{epoch:02d}.eqx"


# A finished run's curriculum may be extended by appending window lengths.
# The extending configuration then differs from the stored run manifest in
# these keys only: the window list, the prose description, the file's digest
# and the commit of the extending code.  Everything else, epochs_per_window
# included, must be identical.
_EXTENSION_KEYS = ("config", "config_sha256", "curriculum", "legoesm_git_sha")


def run_identity(stored: dict) -> dict:
    """A stored run manifest without its record of curriculum extensions."""
    return {key: value for key, value in stored.items()
            if key != "curriculum_extensions"}


def curriculum_extension(stored: dict, identity: dict, log: dict,
                         run_dir: Path) -> dict | None:
    """The extension record when ``identity`` appends window lengths to the
    completed curriculum of the ``stored`` run manifest; ``None`` when the
    two are identical (a plain resume).  Any other difference refuses."""
    if identity == run_identity(stored):
        return None

    def apart_from_the_extension(manifest: dict) -> dict:
        manifest = copy.deepcopy(run_identity(manifest))
        for block in (manifest, manifest["config"]):
            block["curriculum"].pop("window_days")
        manifest["config"].pop("description")
        for key in ("config_sha256", "legoesm_git_sha"):
            manifest.pop(key)
        return manifest

    old = stored["curriculum"]["window_days"]
    new = identity["curriculum"]["window_days"]
    if apart_from_the_extension(identity) != apart_from_the_extension(stored) \
            or new == old:
        differing = sorted(key for key in identity
                           if identity.get(key) != stored.get(key))
        raise RuntimeError(
            f"refusing to resume {run_dir}: the run differs in {differing}")
    if len(new) <= len(old) or new[:len(old)] != old:
        raise RuntimeError(
            f"refusing to resume {run_dir}: window lengths {new} do not "
            f"extend the run's {old} (only appending is admitted)")
    if log["status"] != "complete":
        raise RuntimeError(
            f"refusing to extend {run_dir}: its curriculum {old} has not "
            f"completed (status {log['status']!r}); resume it with its own "
            "configuration first")
    return {
        "extended_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "from_window_days": old, "to_window_days": new,
        "epochs_per_window": stored["curriculum"]["epochs_per_window"],
        "completed_epochs_at_extension": len(log["epochs"]),
        "from_config_sha256": stored["config_sha256"],
        "to_config_sha256": identity["config_sha256"],
        "from_legoesm_git_sha": stored["legoesm_git_sha"],
        "legoesm_git_sha": identity["legoesm_git_sha"]}


def train(config: dict, config_path: Path, *, card, seed: int, run_dir: Path,
          device: dict, windows_per_call: int, commit: str) -> dict:
    """Run, or resume, one seed of one configuration in ``run_dir``."""
    if seed not in config["seeds"]:
        raise ValueError(f"seed {seed} is not one of the configuration's seeds")
    if windows_per_call not in (1, 2, 5, 10):
        raise ValueError("windows_per_call must divide the batch of ten")
    geometry = build_closure_geometry(card)
    spec, scales = network_spec(config, card, geometry)
    settings = SWGOptimizerSettings(**config["optimiser"])
    optimizer = create_swg_optimizer(settings)
    run_dir = Path(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    log_path = run_dir / "training_log.json"
    contract_path = run_dir / "dataset_contract.json"
    invocation = {
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        **{key: device[key] for key in (
            "physical_gpu", "cuda_visible_devices",
            "xla_python_client_mem_fraction")}}
    def identity_for(path: Path) -> dict:
        return _identity(
            config, config_path, card, geometry, spec, scales, _sha256(path),
            json.loads(path.read_text()), seed=seed, device=device,
            windows_per_call=windows_per_call, commit=commit)

    key = jax.random.PRNGKey(seed)
    network = SWGScalarPotentialCNN(spec, key=key)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
    pretrained = config["network"].get("initial_checkpoint")
    if pretrained is not None:
        # The file's network replaces the seed's initialisation; its
        # optimiser state is discarded (the first stage starts fresh).  A
        # structurally different file raises here, before any run directory.
        network, _ = load_checkpoint_or_fail(
            Path(pretrained), (network, opt_state),
            what="network initial_checkpoint")

    if manifest_path.is_file():
        stored = json.loads(manifest_path.read_text())
        log = json.loads(log_path.read_text())
        identity = identity_for(contract_path)
        extension = curriculum_extension(stored, identity, log, run_dir)
        if extension is not None:
            # The finished stages stay as logged; the manifest becomes the
            # extending configuration's and keeps the record of every
            # extension, so the extended run resumes as any other.
            _write_json(manifest_path, stored
                        | {key: identity[key] for key in _EXTENSION_KEYS}
                        | {"curriculum_extensions":
                           stored.get("curriculum_extensions", [])
                           + [extension]})
            log["status"] = "running"
            invocation["curriculum_extension"] = extension
    else:
        if run_dir.exists():
            raise FileExistsError(
                f"{run_dir} exists without a run manifest; remove it or use "
                "another run root")
        # A start whose data does not validate leaves no run directory: the
        # contract is validated and the manifest and an empty log written in
        # a staging directory that becomes the run directory in one rename.
        staging = run_dir.with_name(f".{run_dir.name}.staging-{os.getpid()}")
        staging.mkdir(parents=True)
        try:
            (staging / "checkpoints").mkdir()
            write_dataset_contract(config, staging / contract_path.name)
            SWGDatasetContract.from_json(staging / contract_path.name, card=card)
            identity = identity_for(staging / contract_path.name)
            _write_json(staging / manifest_path.name, identity)
            log = {"schema": LOG_SCHEMA, "run_id": identity["run_id"],
                   "status": "running", "gates": None, "epochs": [],
                   "invocations": [],
                   **{key: identity["network"][key] for key in
                      ("initial_checkpoint",) if key in identity["network"]}}
            _write_json(staging / log_path.name, log)
        except BaseException:
            shutil.rmtree(staging)
            raise
        staging.rename(run_dir)
    contract = SWGDatasetContract.from_json(contract_path, card=card)
    if config["dataset"]["cache"] == "memory":
        loaded = time.perf_counter()
        held = contract.preload_memory(
            window_days=config["curriculum"]["window_days"],
            target_interval_hours=config["target_interval_hours"])
        print(f"DATASET_CACHE hours={held['hours']} bytes={held['bytes']} "
              f"seconds={time.perf_counter() - loaded:.1f}", flush=True)
    log["invocations"].append(invocation | {
        "completed_epochs_at_start": len(log["epochs"])})
    _write_json(log_path, log)

    if log["epochs"]:
        last = log["epochs"][-1]
        checkpoint = run_dir / last["checkpoint"]["path"]
        if _sha256(checkpoint) != last["checkpoint"]["sha256"]:
            raise RuntimeError(f"checkpoint digest differs: {checkpoint}")
        network, opt_state = load_checkpoint_or_fail(
            checkpoint, (network, opt_state), what="SWG closure and optimiser")
    done = {(entry["window_days"], entry["epoch"]) for entry in log["epochs"]}
    windows = config["curriculum"]["window_days"]
    epochs = config["curriculum"]["epochs_per_window"]
    first_step = not log["epochs"]

    for window_days in windows:
        value_and_grad = loss_fn = None
        for epoch in range(1, epochs + 1):
            if (window_days, epoch) in done:
                continue
            if value_and_grad is None:
                value_and_grad, loss_fn = build_batch_functions(
                    card, geometry, window_days=window_days,
                    target_interval_hours=config["target_interval_hours"],
                    objective=config["objective"],
                    wall_ring=config["wall_ring"],
                    training_precision=config["training_precision"],
                    checkpoint=config["gradient_checkpointing"])
                update = _update_function(optimizer)
            if epoch == 1:
                opt_state = optimizer.init(
                    eqx.filter(network, eqx.is_inexact_array))
            if first_step:
                log["gates"] = _start_gates(
                    network, contract, card, geometry, scales,
                    window_days=window_days, seed=seed, key=key,
                    target_interval_hours=config["target_interval_hours"])
                _write_json(log_path, log)
                if not log["gates"]["initial_closure_acceleration"]["passes"]:
                    raise RuntimeError(
                        "initial closure acceleration exceeds 1% of the wind "
                        "acceleration on the first batch's ten starts")
            started = time.perf_counter()
            losses = []
            for step, batch in enumerate(contract.batches(
                    split="train", window_days=window_days,
                    target_interval_hours=config["target_interval_hours"],
                    epoch=epoch - 1, seed=seed)):
                if batch.split != "train" or batch.window_days != window_days:
                    raise RuntimeError("contract issued a mislabelled batch")
                loss, grads = batch_value_and_grad(
                    value_and_grad, network, batch, key, windows_per_call)
                where = f"window {window_days} d, epoch {epoch}, step {step}"
                _require_finite(loss, grads, where)
                if first_step:
                    assert_network_gradients_live(grads)
                    log["gates"]["first_step_no_inert"] = {
                        "passed": True, "leaves": len(jax.tree.leaves(
                            eqx.filter(grads, eqx.is_inexact_array)))}
                    first_step = False
                network, opt_state = update(network, grads, opt_state)
                losses.append(float(loss))
            validation = [
                batch_loss_value(loss_fn, network, batch, key, windows_per_call)
                for batch in contract.batches(
                    split="validation", window_days=window_days,
                    target_interval_hours=config["target_interval_hours"])]
            name = _checkpoint_name(window_days, epoch)
            save_checkpoint((network, opt_state), run_dir / name)
            log["epochs"].append({
                "window_days": window_days, "epoch": epoch,
                "optimizer_steps": len(losses),
                "train_loss_mean": float(np.mean(losses)),
                "train_loss_per_batch": losses,
                "validation_loss": float(np.mean(validation)),
                "validation_windows": TEMPLATE_BATCH_SIZE * len(validation),
                "wall_seconds": time.perf_counter() - started,
                "checkpoint": {"path": name,
                               "sha256": _sha256(run_dir / name)}})
            _write_json(log_path, log)
    log["status"] = "complete"
    _write_json(log_path, log)
    return log


def free_run_output(config: dict, run_dir: Path, window_days: int,
                    output: Path | None) -> Path:
    """``<run_dir>/free_runs/wNN`` from rest, ``wNN_dayDDDD`` from a record
    start, or the explicit ``output`` directory."""
    if output is not None:
        return Path(output)
    start = config["free_run"]["start"]
    suffix = "" if start == "rest" else f"_day{start['day']:04d}"
    return run_dir / "free_runs" / f"w{window_days:02d}{suffix}"


def run_free_run(config: dict, config_path: Path, *, card, seed: int,
                 run_dir: Path, window_days: int, device: dict,
                 commit: str, output: Path | None = None) -> dict:
    """Free-run one trained window length of one seed (``free-run``).

    The run directory must be this configuration file's (digest) and seed's;
    the window length's last epoch must be logged; its checkpoint must match
    the logged digest before it is loaded.  The device and math policy must
    be the configuration's free-run execution.  An incomplete earlier free
    run is archived; a complete one is never overwritten.
    """
    run_dir = Path(run_dir)
    execution = {key: device[key] for key in ("device", "math_library")}
    if execution != {key: config["free_run"][key]
                     for key in ("device", "math_library")}:
        raise RuntimeError(f"free runs execute as configured "
                           f"({config['free_run']['device']}, "
                           f"{config['free_run']['math_library']}), "
                           f"not {execution}")
    identity = json.loads((run_dir / "run_manifest.json").read_text())
    # The free-run block (days, start, device) and the prose description are
    # the free run's own; every other key must be the training configuration
    # the run was made from.
    if config | {key: identity["config"][key]
                 for key in ("free_run", "description")} \
            != identity["config"] or identity["seed"]["value"] != seed:
        raise ValueError(f"{run_dir} is not this configuration's seed {seed}")
    epochs = config["curriculum"]["epochs_per_window"]
    log = json.loads((run_dir / "training_log.json").read_text())
    final = next((entry for entry in log["epochs"]
                  if entry["window_days"] == window_days
                  and entry["epoch"] == epochs), None)
    if final is None:
        raise ValueError(f"window length {window_days} d has not completed "
                         f"its {epochs} epochs in {run_dir}")
    output = free_run_output(config, run_dir, window_days, output)
    if (output / "manifest.json").is_file():
        raise FileExistsError(f"{output} already holds a complete free run")
    geometry = build_closure_geometry(card)
    spec, _ = network_spec(config, card, geometry)
    closure = free_run_closure({
        "kind": "trained",
        "run_id": identity["run_id"], "experiment": config["experiment"],
        "config_sha256": identity["config_sha256"],
        "dataset_contract_sha256": identity["dataset"]["contract_sha256"],
        "network_seed": seed, "window_days": window_days,
        "objective": config["objective"], "wall_ring": config["wall_ring"],
        "network_spec": _jsonable(spec._asdict()),
        "checkpoint_path": str((run_dir / final["checkpoint"]["path"]).resolve()),
        "checkpoint_sha256": final["checkpoint"]["sha256"]})
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(seed))
    optimizer = create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))
    network, _ = load_checkpoint_or_fail(
        closure["checkpoint_path"],
        (network, optimizer.init(eqx.filter(network, eqx.is_inexact_array))),
        what="SWG closure and optimiser")
    _archive_incomplete(output)
    return free_run(network, card, geometry, output,
                    days=config["free_run"]["days"], closure=closure,
                    commit=commit, device=device,
                    start=resolve_start(config, card))


def run_control_free_run(config: dict, *, card, output: Path, device: dict,
                         commit: str) -> dict:
    """Run one explicitly configured no-closure control from rest."""
    if config.get("schema") != CONTROL_CONFIG_SCHEMA:
        raise ValueError("no-closure free run requires a control configuration")
    execution = {key: device[key] for key in ("device", "math_library")}
    configured = {key: config["free_run"][key]
                  for key in ("device", "math_library")}
    if execution != configured:
        raise RuntimeError(
            f"control free run executes as configured {configured}, not {execution}")
    if (output / "manifest.json").is_file():
        raise FileExistsError(f"{output} already holds a complete free run")
    _archive_incomplete(output)
    return free_run(
        None, card, build_closure_geometry(card), output,
        days=config["free_run"]["days"],
        closure=free_run_closure({"kind": "none", "control": config["control"]}),
        commit=commit, device=device, start=resolve_start(config, card))


def _update_function(optimizer):
    @eqx.filter_jit
    def update(network, grads, opt_state):
        params = eqx.filter(network, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return eqx.apply_updates(network, updates), opt_state
    return update


def _start_gates(network, contract, card, geometry, scales, *,
                 window_days: int, seed: int, key,
                 target_interval_hours: int) -> dict:
    """Round-6 acceleration bar and the Eq. 6 report on the first batch."""
    batch = next(contract.batches(
        split="train", window_days=window_days,
        target_interval_hours=target_interval_hours, epoch=0, seed=seed))
    states = batch_start_states(card, batch)
    return {
        "start_absolute_hours": [int(hour) for hour in batch.start_absolute_hour],
        "initial_closure_acceleration": initial_closure_acceleration_ratio(
            network, states, card, geometry, key=key),
        "eq6_scale_report": eq6_scale_report(
            network, states, card, geometry, scales, key=key),
    }


# -------------------------------------------------------------------- profile

def _peak_memory(device: dict) -> dict:
    record = {"peak_rss_kib": int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    if device["device"] == "cuda":
        stats = jax.devices()[0].memory_stats() or {}
        record["device_peak_bytes_in_use"] = int(
            stats.get("peak_bytes_in_use", -1))
    return record


def profile(config: dict, *, card, device: dict, windows_per_call: int) -> dict:
    """Time one optimiser step per window length, batch 10, and one free day.

    Rest-state starts and zero targets: the compiled programs have fixed
    shapes and no data-dependent branches, so their cost does not depend on
    the values.  Extrapolates one seed from the split lengths: training
    windows per epoch ``floor(train_days / w)`` (the contract's lattice
    count at offset zero, an upper bound), the same for validation.  The
    free-run day is timed only on the free runs' configured device.
    """
    geometry = build_closure_geometry(card)
    spec, _ = network_spec(config, card, geometry)
    key = jax.random.PRNGKey(config["seeds"][0])
    network = SWGScalarPotentialCNN(spec, key=key)
    optimizer = create_swg_optimizer(
        SWGOptimizerSettings(**config["optimiser"]))
    update = _update_function(optimizer)
    shape = tuple(geometry.tmask.shape)
    rest = np.zeros((TEMPLATE_BATCH_SIZE,) + shape)
    splits = config["dataset"]["splits_days"]
    epochs = config["curriculum"]["epochs_per_window"]
    rows = []
    for window_days in config["curriculum"]["window_days"]:
        value_and_grad, loss_fn = build_batch_functions(
            card, geometry, window_days=window_days,
            target_interval_hours=config["target_interval_hours"],
            objective=config["objective"], wall_ring=config["wall_ring"],
            training_precision=config["training_precision"],
            checkpoint=config["gradient_checkpointing"])
        target = np.zeros(
            (TEMPLATE_BATCH_SIZE,
             HOURS_PER_DAY * window_days // config["target_interval_hours"])
            + shape)
        batch = WindowBatch((rest,) * 3, (target,) * 3,
                            np.zeros(TEMPLATE_BATCH_SIZE, dtype=np.int64),
                            "profile", window_days)
        opt_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
        timings = {}
        for label in ("first_s", "step_s"):
            started = time.perf_counter()
            loss, grads = batch_value_and_grad(
                value_and_grad, network, batch, key, windows_per_call)
            stepped, _ = update(network, grads, opt_state)
            jax.block_until_ready(jax.tree.leaves(stepped))
            timings[label] = time.perf_counter() - started
        for label in ("forward_first_s", "forward_s"):
            started = time.perf_counter()
            batch_loss_value(loss_fn, network, batch, key, windows_per_call)
            timings[label] = time.perf_counter() - started
        train_batches = (splits["train"][1] - splits["train"][0]) \
            // window_days // TEMPLATE_BATCH_SIZE
        validation_batches = (splits["validation"][1]
                              - splits["validation"][0]) \
            // window_days // TEMPLATE_BATCH_SIZE
        epoch_s = (train_batches * timings["step_s"]
                   + validation_batches * timings["forward_s"])
        rows.append({"window_days": window_days, **timings,
                     "train_batches_per_epoch": train_batches,
                     "validation_batches_per_epoch": validation_batches,
                     "epoch_s": epoch_s, "window_s": epochs * epoch_s,
                     "loss_finite": bool(np.isfinite(float(loss))),
                     **_peak_memory(device)})
        # A longer window may not fit in memory: report each as it lands.
        print(f"PROFILE_WINDOW {json.dumps(rows[-1])}", flush=True)
    # Free runs execute only on their configured device (another process).
    free, excludes = None, "compilation, data loading and rebalancing"
    if device["device"] == config["free_run"]["device"]:
        forcing, one_day = free_run_day(network, card, geometry, key=key)
        state = _double_gyre_driver().seed_steady_forcing_carry(
            card.recipe.initial_state, forcing)
        times = []
        for _ in range(3):
            started = time.perf_counter()
            state, _ = one_day(state)
            jax.block_until_ready(state.eta.data)
            times.append(time.perf_counter() - started)
        day_s = (times[1] + times[2]) / 2.0  # day one carries the compilation
        free = {"seconds_per_day": day_s,
                "seconds_per_run": config["free_run"]["days"] * day_s}
    else:
        excludes += (f"; free runs (configured on {config['free_run']['device']}"
                     ": profile there)")
    training_s = sum(row["window_s"] for row in rows)
    free_s = 0.0 if free is None else len(rows) * free["seconds_per_run"]
    return {
        "schema": PROFILE_SCHEMA, "device": device,
        "windows_per_call": windows_per_call, "batch_size": TEMPLATE_BATCH_SIZE,
        "network_parameter_count": network_parameter_count(network),
        "windows": rows,
        "free_run": free,
        "one_seed_estimate": {
            "training_hours": training_s / 3600.0,
            "free_run_hours": None if free is None else free_s / 3600.0,
            "total_hours": (training_s + free_s) / 3600.0,
            "excludes": excludes},
    }


# ----------------------------------------------------------------------- main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "free-run", "profile", "check-data"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name != "free-run":  # a free run's device is the configuration's
            command.add_argument("--device", choices=("cpu", "cuda"),
                                 required=True)
            command.add_argument("--physical-gpu", type=int)
        if name in ("train", "profile"):
            command.add_argument("--windows-per-call", type=int, required=True,
                                 choices=(1, 2, 5, 10))
        if name == "train":
            command.add_argument("--seed", type=int, required=True)
            command.add_argument("--run-root", type=Path, required=True)
        elif name == "free-run":
            command.add_argument("--seed", type=int)
            command.add_argument("--run-root", type=Path, required=True)
            command.add_argument("--physical-gpu", type=int)
        else:
            command.add_argument("--output", type=Path, required=True)
        if name == "free-run":
            command.add_argument("--window-days", type=int)
            command.add_argument(
                "--output", type=Path,
                help="directory of this trained free run (default "
                     "<run>/free_runs/wNN, or wNN_dayDDDD from a record start)")
    args = parser.parse_args(argv)
    try:
        from legoesm.ocean.fidelity.provenance import git_sha

        config = load_config(args.config)
        if args.command == "free-run":
            device = select_device(
                config["free_run"]["device"], config["free_run"]["math_library"],
                args.physical_gpu, reproducible_cuda=True)
        else:
            math_library = "native" if args.device == "cuda" else "libm"
            device = select_device(args.device, math_library, args.physical_gpu)
        card = build_deck_card(config["deck"], config["surface"])
        if args.command == "free-run":
            if config["schema"] == CONTROL_CONFIG_SCHEMA:
                if args.seed is not None or args.window_days is not None \
                        or args.output is not None:
                    raise ValueError(
                        "no-closure control free runs do not take --seed, "
                        "--window-days or --output")
                output = args.run_root / config["experiment"]
                manifest = run_control_free_run(
                    config, card=card, output=output, device=device,
                    commit=git_sha())
            else:
                if args.seed is None or args.window_days is None:
                    raise ValueError(
                        "trained free runs require --seed and --window-days")
                run_dir = (args.run_root / config["experiment"]
                           / f"seed{args.seed}")
                manifest = run_free_run(
                    config, args.config, card=card, seed=args.seed,
                    run_dir=run_dir, window_days=args.window_days, device=device,
                    commit=git_sha(), output=args.output)
                output = free_run_output(
                    config, run_dir, args.window_days, args.output)
            print(f"FREE_RUN_COMPLETE root={output.resolve()} "
                  f"manifest_sha256={_sha256(output / 'manifest.json')} "
                  f"all_finite={manifest['numerical_health']['all_finite']}")
            return 0
        if args.command == "check-data":
            report = check_data(config, card=card)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            _write_json(args.output, report | {"device": device})
            print(f"DATA_CHECKED sources={report['source_count']} "
                  f"batches={len(report['batches'])} "
                  f"contract_sha256={report['contract_sha256']}")
            return 0
        if args.command == "train":
            run_dir = args.run_root / config["experiment"] / f"seed{args.seed}"
            train(config, args.config, card=card, seed=args.seed,
                  run_dir=run_dir, device=device,
                  windows_per_call=args.windows_per_call, commit=git_sha())
            print(f"TRAINING_COMPLETE run={run_dir.resolve()}")
            return 0
        report = profile(config, card=card, device=device,
                         windows_per_call=args.windows_per_call)
        report["legoesm_git_sha"] = git_sha(allow_dirty=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.output, report)
        estimate = report["one_seed_estimate"]
        print(f"PROFILE device={device['device']} "
              f"step_s={[round(row['step_s'], 3) for row in report['windows']]} "
              f"one_seed_hours={estimate['total_hours']:.2f} "
              f"output={args.output.resolve()}")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
