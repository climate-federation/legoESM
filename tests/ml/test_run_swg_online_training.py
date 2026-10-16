"""Direct tests for the SWG online training driver (experiment one)."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import shutil
import types
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ml.checkpoint_io import CheckpointStructureError
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_swg_card,
    swg_surface_wind,
)
from legoesm.ocean.state import OceanSurfaceForcing
from legoesm.training.swg_online import (
    DAILY_MEAN_FIELDS,
    SNAPSHOT_FIELDS,
    SWGOptimizerSettings,
    SWGScalarPotentialCNN,
    build_closure_geometry,
    build_segment_fn,
    coarse_state_from_native,
    create_swg_optimizer,
    deterministic_trajectory_loss,
    maddison_eq6_network_scales,
    swg_online_step,
)

_ROOT = Path(__file__).parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


driver = _load("run_swg_online_training",
               _ROOT / "scripts/run/run_swg_online_training.py")
# One synthetic dual-record writer for the lane: the dataset-contract tests'.
records = _load("swg_online_record_writer",
                Path(__file__).with_name("test_swg_online.py"))
evaluation = _load(
    "swg_online_evaluation", _ROOT / "scripts/validate/ocean_fidelity/"
    "testcases/nemo_testcase_l1_swg_online_evaluation.py")
CONFIGS = sorted((_ROOT / "config/swg_online").glob("exp1_*.json"))
PILOT = _ROOT / "config/swg_online/pilot_template_ring_excluded.json"
CONTROL_CONFIGS = {
    name: _ROOT / f"config/swg_online/control_{name}_gpu.json"
    for name in ("nu10", "nu80")}
SMOKE_CONFIG = _ROOT / "config/swg_online/control_nu10_gpu_smoke_2d.json"
CPU = {"device": "cpu", "backend": "cpu", "precision_policy": "fp64",
       "math_library": "libm", "physical_gpu": None,
       "cuda_visible_devices": None, "jax_platforms": "cpu",
       "xla_flags": "", "jax_version": jax.__version__, "gpu_name": None,
       "xla_python_client_mem_fraction": None}


@pytest.fixture(scope="module", autouse=True)
def _fp64():
    old = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    jax.config.update("jax_enable_x64", True)
    yield
    set_policy(old)


# ----------------------------------------------------------- configurations

def test_committed_configs_are_the_three_candidates_of_the_template_procedure():
    loaded = {path.name: driver.load_config(path) for path in CONFIGS}
    assert {(config["objective"], config["wall_ring"])
            for config in loaded.values()} == {
        ("template", "excluded"), ("template", "weighted"),
        ("symmetric", "weighted")}
    assert len({config["experiment"] for config in loaded.values()}) == 3
    for name, config in loaded.items():
        assert config["objective"] in name and config["wall_ring"] in name
        assert config["experiment"] == "swgonl_exp1_" + name[5:-5]
        assert config["curriculum"] == {
            "window_days": [1, 2, 4, 8], "epochs_per_window": 5,
            "network_between_windows": "carried",
            "optimiser_between_windows": "reinitialised"}
        # Maddison 2026: AdamW with the Keras defaults at a constant rate.
        assert config["optimiser"] == {
            "optimizer": "adamw", "learning_rate": 1.0e-3,
            "schedule": "constant", "weight_decay": 0.004, "adam_b1": 0.9,
            "adam_b2": 0.999, "adam_epsilon": 1.0e-7,
            "adam_epsilon_convention": "keras_epsilon_hat",
            "gradient_clipping": "none"}
        splits = config["dataset"]["splits_days"]
        record_days = splits["validation"][1] - splits["train"][0]
        assert record_days == 12 * 365
        assert splits["train"][1] - splits["train"][0] == 0.8 * record_days
        assert splits["validation"][0] - splits["train"][1] == 30
        assert splits["test"][0] - splits["validation"][1] == 30
        assert config["dataset"]["batch_size"] == 10
        assert len(config["seeds"]) >= 3
        assert config["network"] == {
            "hidden_channels": 16, "hidden_layers": 4, "kernel_size": 3,
            "scales": driver.SCALE_RULE}
        # Daily means beside the snapshots.  The candidate chosen for the full
        # experiment (template, wall ring weighted) runs its free runs on the
        # GPU with native math (user decision 2026-10-14, after the GPU
        # controls passed the device check); the other candidates keep the
        # controls' CPU and libm execution.
        chosen = (config["objective"], config["wall_ring"]) == ("template", "weighted")
        assert config["free_run"] == {
            "days": 7300, "start": "rest",
            "sampling": "end_of_day_snapshot_and_daily_mean",
            "device": "cuda" if chosen else "cpu",
            "math_library": "native" if chosen else "libm"}
        assert config["deck"] == "dg50"


def test_pilot_restricts_seeds_windows_and_training_years_by_its_own_keys():
    """The pilot is candidate template+excluded with only these keys changed:
    one seed, window lengths 1 and 2 days, and the train split cut to the
    first two training years of the 12-year record; validation, test and
    every other choice are experiment one's."""
    pilot = driver.load_config(PILOT)
    base = driver.load_config(
        _ROOT / "config/swg_online/exp1_template_ring_excluded.json")
    first = base["dataset"]["splits_days"]["train"][0]
    expected = copy.deepcopy(base)
    expected.update(experiment="swgonl_pilot_template_ring_excluded",
                    description=pilot["description"], seeds=[20261013])
    expected["dataset"]["splits_days"]["train"] = [first, first + 2 * 365]
    expected["curriculum"]["window_days"] = [1, 2]
    expected["free_run"].update(device="cuda", math_library="native")
    assert pilot == expected
    assert pilot["dataset"]["splits_days"]["train"] == [731, 1461]


def test_gpu_control_configs_name_the_frozen_control_choices_exactly():
    for name, path in CONTROL_CONFIGS.items():
        config = driver.load_config(path)
        assert config == {
            "schema": driver.CONTROL_CONFIG_SCHEMA,
            "experiment": f"swgonl_control_{name}_gpu",
            "description": config["description"],
            "control": name,
            "deck": "dg50" if name == "nu10" else "dg50-nu80",
            "closure": "none",
            "free_run": {
                "days": 7300, "start": "rest",
                "sampling": "end_of_day_snapshot_and_daily_mean",
                "device": "cuda", "math_library": "native"},
        }
    smoke = driver.load_config(SMOKE_CONFIG)
    assert smoke["control"] == "nu10"
    assert smoke["free_run"] == {
        "days": 2, "start": "rest",
        "sampling": "end_of_day_snapshot_and_daily_mean",
        "device": "cuda", "math_library": "native"}


@pytest.mark.parametrize("edit,match", [
    (lambda c: c.pop("objective"), "exactly the keys"),
    (lambda c: c.pop("wall_ring"), "exactly the keys"),
    (lambda c: c.update(extra=1), "exactly the keys"),
    (lambda c: c.update(objective="auto"), "objective"),
    (lambda c: c.update(wall_ring=None), "wall_ring"),
    (lambda c: c["optimiser"].pop("adam_epsilon_convention"), "exactly"),
    (lambda c: c["optimiser"].update(schedule="warmup_cosine"), "constant"),
    (lambda c: c["optimiser"].update(gradient_clipping="global_norm"), "none"),
    (lambda c: c["optimiser"].update(learning_rate=1), "finite float"),
    (lambda c: c.update(seeds=[]), "one or more distinct"),
    (lambda c: c.update(seeds=[1, 1, 2]), "one or more distinct"),
    (lambda c: c["curriculum"].update(window_days=[1, 16]), "window_days"),
    (lambda c: c["curriculum"].update(window_days=[2, 1]), "window_days"),
    (lambda c: c["curriculum"].update(optimiser_between_windows="carried"),
     "reinitialised"),
    (lambda c: c["dataset"].update(batch_size=8), "batch_size"),
    (lambda c: c["network"].update(scales="unit"), "scales"),
    (lambda c: c["network"].update(initial_checkpoint="pretrained.eqx"),
     "absolute path"),
    (lambda c: c["free_run"].update(start="restart"), "rest"),
    (lambda c: c["free_run"].update(sampling="daily_snapshot"), "sampling"),
    (lambda c: c["free_run"].pop("device"), "exactly the keys"),
    (lambda c: c["free_run"].update(device="tpu"), r"cpu\+libm"),
    (lambda c: c["free_run"].update(math_library="fast"), r"cpu\+libm"),
    (lambda c: c.update(deck="maddison"), "deck"),
])
def test_config_refuses_missing_extra_and_unknown_choices(tmp_path, edit, match):
    config = json.loads(CONFIGS[0].read_text())
    edit(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=match):
        driver.load_config(path)


def test_driver_refuses_to_start_without_a_configuration_file(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        driver.main(["train", "--device", "cpu", "--windows-per-call", "10",
                     "--seed", "20261013", "--run-root", str(tmp_path)])
    assert stopped.value.code == 2
    assert "--config" in capsys.readouterr().err
    assert driver.main([
        "train", "--config", str(tmp_path / "absent.json"), "--device",
        "cpu", "--windows-per-call", "10", "--seed", "20261013",
        "--run-root", str(tmp_path)]) == 2
    assert "REFUSE" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_device_selection_is_explicit_and_never_inferred(monkeypatch):
    with pytest.raises(RuntimeError, match="exactly 'cuda'"):
        driver.select_device("cuda", "native", 1, environ={
            "JAX_PLATFORMS": "gpu", "CUDA_VISIBLE_DEVICES": "1"})
    with pytest.raises(RuntimeError, match="exactly 'cuda'"):
        driver.select_device("cuda", "native", 1,
                             environ={"CUDA_VISIBLE_DEVICES": "1"})
    with pytest.raises(ValueError, match="--physical-gpu"):
        driver.select_device("cuda", "native", None,
                             environ={"JAX_PLATFORMS": "cuda"})
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        driver.select_device("cuda", "native", 1, environ={
            "JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": "0"})
    with pytest.raises(ValueError, match="only valid"):
        driver.select_device("cpu", "libm", 0, environ={"JAX_PLATFORMS": "cpu"})
    with pytest.raises(RuntimeError, match="exactly 'cpu'"):
        driver.select_device("cpu", "libm", None, environ={})
    with pytest.raises(RuntimeError, match="selected backend"):
        driver.select_device("cuda", "native", 1, environ={
            "JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": "1"})
    with pytest.raises(ValueError, match=r"cpu\+libm"):
        driver.select_device("cuda", "libm", 1, environ={})
    previous = get_policy()
    try:
        assert driver.select_device(
            "cpu", "libm", None,
            environ={"JAX_PLATFORMS": "cpu"})["math_library"] \
            == "libm"
        monkeypatch.setattr(driver.jax, "default_backend", lambda: "gpu")
        monkeypatch.setattr(driver.jax, "devices", lambda: [
            types.SimpleNamespace(device_kind="V100S")])
        environment = {"JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": "1",
                       "XLA_FLAGS": "--another_flag=true"}
        selected = driver.select_device(
            "cuda", "native", 1, reproducible_cuda=True, environ=environment)
        assert selected["math_library"] == "native"
        assert selected["physical_gpu"] == 1
        assert selected["gpu_name"] == "V100S"
        assert environment["XLA_FLAGS"].split()[-1] == driver.CUDA_AUTOTUNE_FLAG
        with pytest.raises(RuntimeError, match="conflicting"):
            driver.select_device(
                "cuda", "native", 1, reproducible_cuda=True,
                environ={"JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": "1",
                         "XLA_FLAGS": "--xla_gpu_autotune_level=2"})
    finally:
        set_policy(previous)


# -------------------------------------------------------------------- dataset

# The dataset tests' 1,000 km card (20x20 wet cells), stepped hourly with
# gravity reduced so the RK3 gravity-wave argument stays below sqrt(3)
# (1.0 here): 24 steps a day instead of 216.  The record is synthetic; the
# driver's mechanics, not the deck's physics, are under test.
_SMALL = records._SMALL_NAMELIST._replace(
    rn_Dt_s=3600.0, rn_modified_grav_m_s2=0.05, nn_itend=25_920, nn_write=24,
    nn_stock=25_920)


@pytest.fixture(scope="module")
def small_card():
    return build_swg_card(_SMALL)


def _record(root: Path, card, chunks) -> None:
    geometry = build_closure_geometry(card)
    masks = (np.asarray(geometry.umask), np.asarray(geometry.vmask))
    for first, days in chunks:
        records._write_chunk(
            root / f"dual_day{first:04d}_to_day{first + days:04d}",
            source_day=first, days=days, masks=masks,
            truth_run=root.parent / "truth")


def _config(record_root: Path, **changes) -> dict:
    """The committed network on a 1,000 km card, two windows of two epochs.

    Seed 11 passes the 1% initial-acceleration bar on this record (0.46%);
    smaller networks and most seeds do not on a basin a quarter the size.
    """
    config = json.loads(CONFIGS[0].read_text())
    config.update(
        experiment="test_small", objective="template", wall_ring="excluded",
        seeds=[11, 7, 4])
    config["free_run"]["days"] = 2   # day 1's sampling check reads day 2
    config["dataset"]["records"] = [
        {"root": str(record_root), "chunk_label": "dual"}]
    config["dataset"]["splits_days"] = {
        "train": [0, 22], "validation": [52, 72], "test": [102, 122]}
    config["curriculum"].update(window_days=[1, 2], epochs_per_window=2)
    config.update(changes)
    return config


def _write_config(path: Path, config: dict) -> Path:
    path.write_text(json.dumps(config, indent=1))
    return driver.load_config(path) and path


@pytest.fixture(scope="module")
def record(tmp_path_factory, small_card):
    root = tmp_path_factory.mktemp("swg_record") / "record"
    _record(root, small_card, ((0, 11), (11, 11), (52, 20), (102, 20)))
    return root


def test_contract_pins_only_complete_chunks_that_touch_a_split(
        tmp_path, small_card, record):
    root = tmp_path / "record"
    root.mkdir()
    for name in sorted(path.name for path in record.iterdir()):
        (root / name).symlink_to(record / name)
    _record(root, small_card, ((200, 5),))                 # outside the splits
    pending = root / "dual_day0112_to_day0122"                # still running
    pending.mkdir()
    (pending / "manifest.pending.json").write_text(json.dumps({
        "status": "pending", "days": 10, "completed_days": 3,
        "restart": {"source_day": 112}}))
    (root / "dual_day0011_to_day0022.interrupted").mkdir()  # archived
    config = _config(root)
    payload = driver.write_dataset_contract(config, tmp_path / "contract.json")
    pinned = sorted(Path(source["root"]).name for source in payload["sources"])
    assert pinned == ["dual_day0000_to_day0011", "dual_day0011_to_day0022",
                      "dual_day0052_to_day0072", "dual_day0102_to_day0122"]
    assert payload["splits"] == {"train": [0, 528], "validation": [1248, 1728],
                                 "test": [2448, 2928]}
    assert payload["objective"] == "template"
    manifest = json.loads((root / "dual_day0200_to_day0205/manifest.json")
                          .read_text())
    manifest["days"] = 4
    (root / "dual_day0200_to_day0205/manifest.json").write_text(
        json.dumps(manifest))
    with pytest.raises(ValueError, match="label differs"):
        driver.write_dataset_contract(config, tmp_path / "contract.json")


def test_check_data_reads_every_split_at_every_window_length(
        small_card, record):
    config = _config(record)
    report = driver.check_data(config, card=small_card)
    assert [(row["split"], row["window_days"]) for row in report["batches"]] \
        == [(split, window) for window in (1, 2)
            for split in ("train", "validation", "test")]
    for row in report["batches"]:
        start, end = report["splits_hours"][row["split"]]
        hours = np.asarray(row["first_batch_start_hours"])
        assert hours.size == 10 and hours.min() >= start
        assert hours.max() + 24 * row["window_days"] <= end
        if row["split"] != "train":       # held out: the fixed lattice
            assert list(hours) == list(
                start + 24 * row["window_days"] * np.arange(10))
    assert report["source_count"] == 4
    uncovered = _config(record)
    uncovered["dataset"]["splits_days"]["test"] = [102, 130]
    with pytest.raises(ValueError, match="test split"):
        driver.check_data(uncovered, card=small_card)


def test_a_start_whose_data_does_not_validate_leaves_no_run_directory(
        tmp_path, small_card, record):
    config = _config(record)
    config["dataset"]["splits_days"]["test"] = [102, 130]   # not generated
    path = _write_config(tmp_path / "config.json", config)
    run_dir = tmp_path / "runs" / "seed11"
    with pytest.raises(ValueError, match="test split"):
        driver.train(config, path, card=small_card, seed=11, run_dir=run_dir,
                     device=CPU, windows_per_call=10, commit="0" * 40)
    assert not (tmp_path / "runs").exists() or not any(
        (tmp_path / "runs").iterdir())


def test_a_start_killed_after_the_run_appears_leaves_a_resumable_run(
        tmp_path, small_card, record, monkeypatch):
    config = _config(record)
    path = _write_config(tmp_path / "config.json", config)
    original = driver.SWGDatasetContract.from_json
    calls = []

    def killed_after_rename(contract_path, *, card):
        calls.append(contract_path)
        if len(calls) == 2:          # the first call validates in staging
            raise KeyboardInterrupt("killed after the run directory appeared")
        return original(contract_path, card=card)

    monkeypatch.setattr(driver.SWGDatasetContract, "from_json",
                        staticmethod(killed_after_rename))
    with pytest.raises(KeyboardInterrupt):
        driver.train(config, path, card=small_card, seed=11,
                     run_dir=tmp_path / "run", device=CPU,
                     windows_per_call=10, commit="0" * 40)
    run = tmp_path / "run"
    assert (run / "run_manifest.json").is_file()
    assert json.loads((run / "training_log.json").read_text())["epochs"] == []


# ---------------------------------------------------------------- end to end

def _params(path: Path, card, config, seed):
    geometry = build_closure_geometry(card)
    spec, _ = driver.network_spec(config, card, geometry)
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(seed))
    optimizer = create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))
    skeleton = (network, optimizer.init(eqx.filter(network, eqx.is_inexact_array)))
    loaded = driver.load_checkpoint_or_fail(path, skeleton)
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(
        eqx.filter(loaded[0], eqx.is_inexact_array))]
    count = [int(np.asarray(leaf)) for leaf in jax.tree.leaves(loaded[1])
             if np.asarray(leaf).dtype == np.int32 and np.ndim(leaf) == 0]
    initial = [np.asarray(leaf) for leaf in jax.tree.leaves(
        eqx.filter(network, eqx.is_inexact_array))]
    return leaves, count, initial


@pytest.fixture(scope="module")
def trained(tmp_path_factory, small_card, record):
    """A run killed while writing its second checkpoint, then resumed.

    The state the killed process was writing is the uninterrupted result of
    the second epoch; the resumed run recomputes that epoch from the first
    checkpoint and must reproduce it.
    """
    folder = tmp_path_factory.mktemp("swg_train")
    config = _config(record)
    path = _write_config(folder / "config.json", config)
    save = driver.save_checkpoint
    written = []

    def killed_on_second_save(pytree, target):
        if written:
            stream = io.BytesIO()
            eqx.tree_serialise_leaves(stream, pytree)
            written.append(hashlib.sha256(stream.getvalue()).hexdigest())
            raise KeyboardInterrupt("killed while writing checkpoint two")
        written.append(target)
        return save(pytree, target)

    driver.save_checkpoint = killed_on_second_save
    try:
        with pytest.raises(KeyboardInterrupt):
            driver.train(config, path, card=small_card, seed=11,
                         run_dir=folder / "run", device=CPU,
                         windows_per_call=10, commit="0" * 40)
    finally:
        driver.save_checkpoint = save
    interrupted = json.loads((folder / "run/training_log.json").read_text())
    log = driver.train(
        config, path, card=small_card, seed=11, run_dir=folder / "run",
        device=CPU, windows_per_call=10, commit="0" * 40)
    return {"folder": folder, "config": config, "path": path, "log": log,
            "interrupted": interrupted, "killed_epoch_two_sha256": written[1]}


def test_two_window_lengths_train_log_gate_and_change_the_loss(
        trained, small_card):
    log, run = trained["log"], trained["folder"] / "run"
    assert log["status"] == "complete"
    assert [(entry["window_days"], entry["epoch"]) for entry in log["epochs"]] \
        == [(1, 1), (1, 2), (2, 1), (2, 2)]
    losses = [loss for entry in log["epochs"]
              for loss in entry["train_loss_per_batch"]]
    assert len(losses) == 6 and all(np.isfinite(losses))
    # Same fixed validation windows, parameters moved by the updates.
    validation = {(entry["window_days"], entry["epoch"]):
                  entry["validation_loss"] for entry in log["epochs"]}
    assert validation[(1, 2)] < validation[(1, 1)]
    assert validation[(2, 2)] < validation[(2, 1)]
    for entry in log["epochs"]:
        assert np.isfinite(entry["validation_loss"])
        assert entry["validation_windows"] >= 10
        assert entry["checkpoint"]["sha256"] == driver._sha256(
            run / entry["checkpoint"]["path"])
    gates = log["gates"]
    assert gates["initial_closure_acceleration"]["passes"]
    assert len(gates["initial_closure_acceleration"]["per_state_ratio"]) == 10
    assert gates["first_step_no_inert"] == {"passed": True, "leaves": 9}
    assert gates["eq6_scale_report"]["states"] == 10
    manifest = json.loads((run / "run_manifest.json").read_text())
    geometry = build_closure_geometry(small_card)
    assert manifest["network"]["scales"] == maddison_eq6_network_scales(
        small_card, geometry)._asdict()
    assert manifest["optimiser"]["settings"] == trained["config"]["optimiser"]
    assert manifest["loss"]["objective"] == "template"
    assert manifest["loss"]["wall_ring"] == "excluded"
    assert manifest["dataset"]["operator"] == "filtered"


def test_logged_first_loss_is_eq7_of_the_first_batch_recomputed(
        trained, small_card):
    """Window by window, outside the driver: the configured objective and
    wall ring, the contract's first batch and the seed's network."""
    config, run = trained["config"], trained["folder"] / "run"
    contract = driver.SWGDatasetContract.from_json(
        run / "dataset_contract.json", card=small_card)
    batch = next(contract.batches(split="train", window_days=1, epoch=0,
                                  seed=11))
    geometry = build_closure_geometry(small_card)
    spec, _ = driver.network_spec(config, small_card, geometry)
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(11))
    _, segment = build_segment_fn(small_card, geometry, window_days=1)
    noise = jnp.zeros((24, 3) + tuple(geometry.tmask.shape))
    hourly = [segment(network, coarse_state_from_native(
        small_card.recipe.initial_state, *(value[k] for value in batch.initial)),
        noise, jax.random.PRNGKey(11), jnp.asarray(0.0))[1] for k in range(10)]
    predicted = tuple(jnp.stack([window[index] for window in hourly])
                      for index in range(3))
    namelist = small_card.resolved_namelist
    expected = float(deterministic_trajectory_loss(
        predicted, tuple(jnp.asarray(value) for value in batch.target),
        geometry, objective=config["objective"],
        beta_m1_s1=namelist.rn_beta_m_s, basin_length_m=namelist.rn_domsiz_m,
        wall_ring=config["wall_ring"]))
    logged = trained["log"]["epochs"][0]["train_loss_per_batch"][0]
    assert logged == pytest.approx(expected, rel=1.0e-9)


def test_a_failing_start_gate_stops_the_run_before_any_update(
        tmp_path, small_card, record):
    config = _config(record, seeds=[0, 11, 7])   # seed 0: 5.0% on this record
    path = _write_config(tmp_path / "config.json", config)
    with pytest.raises(RuntimeError, match="initial closure acceleration"):
        driver.train(config, path, card=small_card, seed=0,
                     run_dir=tmp_path / "run", device=CPU,
                     windows_per_call=10, commit="0" * 40)
    log = json.loads((tmp_path / "run/training_log.json").read_text())
    assert log["epochs"] == [] and not any(
        (tmp_path / "run/checkpoints").iterdir())
    assert log["gates"]["initial_closure_acceleration"]["maximum_ratio"] > 0.01


@pytest.mark.parametrize("loss,gradient,error,match", [
    (np.nan, 0.5, RuntimeError, "non-finite loss"),
    (1.0, np.inf, RuntimeError, "non-finite loss or gradient"),
    (1.0, 0.0, ValueError, "identically-zero gradient"),
])
def test_first_step_gradient_gates_stop_the_run(
        tmp_path, small_card, record, monkeypatch, loss, gradient, error,
        match):
    config = _config(record)
    path = _write_config(tmp_path / "config.json", config)

    def planted(value_and_grad, network, batch, key, windows_per_call):
        grads = jax.tree.map(lambda leaf: jnp.full_like(leaf, gradient),
                             eqx.filter(network, eqx.is_inexact_array))
        return jnp.asarray(loss), grads

    monkeypatch.setattr(driver, "batch_value_and_grad", planted)
    with pytest.raises(error, match=match):
        driver.train(config, path, card=small_card, seed=11,
                     run_dir=tmp_path / "run", device=CPU,
                     windows_per_call=10, commit="0" * 40)
    assert json.loads((tmp_path / "run/training_log.json").read_text())[
        "epochs"] == []


def test_batch_parts_average_to_the_batch_loss_and_gradient():
    rng = np.random.default_rng(9)
    batch = driver.WindowBatch(
        tuple(rng.normal(size=(10, 3, 4)) for _ in range(3)),
        tuple(rng.normal(size=(10, 24, 3, 4)) for _ in range(3)),
        np.arange(10), "train", 1)

    def per_window_mean(network, initial, target, key):
        return (jnp.mean(target[1] * network["a"]),
                {"a": jnp.mean(initial[2] * target[0][:, 0], axis=0)})

    reference = driver.batch_value_and_grad(
        per_window_mean, {"a": 2.0}, batch, None, 10)
    for windows_per_call in (1, 2, 5):
        loss, grads = driver.batch_value_and_grad(
            per_window_mean, {"a": 2.0}, batch, None, windows_per_call)
        assert float(loss) == pytest.approx(float(reference[0]), rel=1.0e-13)
        np.testing.assert_allclose(grads["a"], reference[1]["a"],
                                   rtol=1.0e-12, atol=1.0e-15)
        assert driver.batch_loss_value(
            lambda *args: per_window_mean(*args)[0], {"a": 2.0}, batch, None,
            windows_per_call) == pytest.approx(float(reference[0]), rel=1.0e-13)


def test_optimiser_is_reinitialised_and_network_carried_between_windows(
        trained, small_card):
    run, config = trained["folder"] / "run", trained["config"]
    steps = {(entry["window_days"], entry["epoch"]): entry["optimizer_steps"]
             for entry in trained["log"]["epochs"]}
    end_one, count_one, initial = _params(
        run / "checkpoints/w01_e02.eqx", small_card, config, 11)
    start_two, count_two, _ = _params(
        run / "checkpoints/w02_e01.eqx", small_card, config, 11)
    assert count_one and set(count_one) == {steps[(1, 1)] + steps[(1, 2)]}
    assert set(count_two) == {steps[(2, 1)]}       # counted from zero again
    # Adam's first step moves each weight by at most the rate (1e-3) plus
    # its decay: window two starts from window one's network, not the seed.
    lr = config["optimiser"]["learning_rate"]
    assert all(np.max(np.abs(new - old)) <= 1.01 * lr
               for new, old in zip(start_two, end_one, strict=True))
    assert any(np.max(np.abs(new - old)) > 1.5 * lr
               for new, old in zip(start_two, initial, strict=True))


def test_resume_reproduces_the_next_update_bit_for_bit(trained):
    log, interrupted = trained["log"], trained["interrupted"]
    assert [(entry["window_days"], entry["epoch"])
            for entry in interrupted["epochs"]] == [(1, 1)]
    assert interrupted["epochs"][0] == log["epochs"][0]
    assert [entry["completed_epochs_at_start"]
            for entry in log["invocations"]] == [0, 1]
    # Network and optimiser state after the resumed epoch two equal, word
    # for word, what the killed process had computed and was writing.
    resumed = next(entry for entry in log["epochs"]
                   if (entry["window_days"], entry["epoch"]) == (1, 2))
    assert resumed["checkpoint"]["sha256"] \
        == trained["killed_epoch_two_sha256"]


def test_free_runs_are_scored_by_the_harness_and_identify_the_closure(
        trained, small_card):
    """Each trained window length's free run (``free-run``, after training)
    names that window's last checkpoint and is admitted by the held-out
    scorer: the manifest binds and day 1 is shown from the data to be an
    end-of-day snapshot (against day 2's mean)."""
    log, config, path = trained["log"], trained["config"], trained["path"]
    run_dir = trained["folder"] / "run"

    def free(window_days):
        return driver.run_free_run(
            config, path, card=small_card, seed=11, run_dir=run_dir,
            window_days=window_days, device=CPU, commit="0" * 40)

    admitted = {}
    for window_days in (1, 2):
        manifest = free(window_days)
        root = run_dir / "free_runs" / f"w{window_days:02d}"
        final = next(entry for entry in log["epochs"]
                     if entry["window_days"] == window_days
                     and entry["epoch"] == 2)
        closure = manifest["closure"]
        assert closure["checkpoint_sha256"] == final["checkpoint"]["sha256"]
        assert closure["checkpoint_path"] == str(
            (run_dir / final["checkpoint"]["path"]).resolve())
        assert closure["network_seed"] == 11
        assert closure["window_days"] == window_days
        assert closure["objective"] == "template"
        assert closure["run_id"] == "test_small/seed11"
        assert manifest["start"] == "rest" and manifest["status"] == "complete"
        admitted[window_days] = evaluation.admit_trained_run(
            root, driver._sha256(root / "manifest.json"),
            objective="template", window_days=window_days)
        ssh, u, v = evaluation._load_native_window(root, 1, 1)
        assert ssh.shape[0] == 1 and np.all(np.isfinite(ssh))
        assert np.any(u != 0.0)
    assert admitted[1]["checkpoint_sha256"] != admitted[2]["checkpoint_sha256"]
    with pytest.raises(FileExistsError, match="complete free run"):
        free(1)
    with pytest.raises(ValueError, match="has not completed"):
        free(4)


def test_end_to_end_two_steps_free_run_and_the_scorers_admission(
        tmp_path, small_card, record, monkeypatch):
    """Round 8 on the tiny synthetic record: two optimiser steps, a CPU/libm
    free run of four days, and the held-out scorer's admission of it on a
    shortened window (days 1-3).  (1) One manifest schema: the driver's
    manifest is admitted, corrupted ones are refused.  (2) Every day file
    holds the end-of-day snapshot beside the mean of every post-step state,
    under the scorer's dual labels, and the free run executes only as the
    configuration states.  (3) The checkpoint file the manifest names is
    checked against its digest."""
    config = _config(record, seeds=[11])
    config["curriculum"].update(window_days=[1], epochs_per_window=1)
    config["free_run"]["days"] = 4
    path = _write_config(tmp_path / "config.json", config)
    run_dir = tmp_path / "run"
    log = driver.train(config, path, card=small_card, seed=11, run_dir=run_dir,
                       device=CPU, windows_per_call=10, commit="0" * 40)
    assert [entry["optimizer_steps"] for entry in log["epochs"]] == [2]
    checkpoint = run_dir / log["epochs"][0]["checkpoint"]["path"]

    def free(device):
        return driver.run_free_run(
            config, path, card=small_card, seed=11, run_dir=run_dir,
            window_days=1, device=device, commit="0" * 40)

    # (2) Not on another device or math library than the configuration's.
    for device in (dict(CPU, device="cuda", backend="gpu",
                        math_library="native", gpu_name="test GPU"),
                   dict(CPU, math_library="native")):
        with pytest.raises(RuntimeError, match="execute as configured"):
            free(device)
    manifest = free(CPU)
    root = run_dir / "free_runs" / "w01"
    sha = driver._sha256(root / "manifest.json")
    assert (manifest["backend"], manifest["math_library"]) == ("cpu", "libm")

    # (1)+(3) The scorer admits the driver's manifest and its checkpoint file.
    admitted = evaluation.admit_trained_run(root, sha, objective="template",
                                            window_days=1)
    assert admitted["checkpoint_path"] == str(checkpoint.resolve())
    assert admitted["checkpoint_sha256"] == driver._sha256(checkpoint) \
        == log["epochs"][0]["checkpoint"]["sha256"]
    assert admitted["network_seed"] == 11

    # (2) Shortened window, days 1-3: dual files, each snapshot shown from the
    # data against its own and the next day's mean (day 4's for day 3).
    loaded = evaluation._load_native_window(root, 1, 3)
    from legoesm.ocean.fidelity.swg_coarse_graining import cgrid_geometry

    monkeypatch.setattr(evaluation, "_dg50", lambda: (
        small_card, cgrid_geometry(_SMALL)))
    reference = evaluation.window_statistics(loaded)
    (score,) = evaluation.score_free_run(
        root, expected_manifest_sha256=sha, reference=reference,
        windows=[(1, 3)], expected_namelist=_SMALL)
    assert score["physical_health"]["passes"]
    assert all(np.isfinite(score[key]) for key in evaluation.SCORE_METRICS)
    with np.load(root / "day0001.npz") as saved:
        assert all(np.array_equal(saved[key], value[0])
                   for key, value in zip(SNAPSHOT_FIELDS, loaded, strict=True))
        stored = [np.asarray(saved[key]) for key in DAILY_MEAN_FIELDS]
    # ... and day 1's mean is that of all 24 post-step states, recomputed
    # one step at a time from the checkpoint the manifest names.
    geometry = build_closure_geometry(small_card)
    spec, _ = driver.network_spec(config, small_card, geometry)
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(11))
    optimizer = create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))
    network, _ = driver.load_checkpoint_or_fail(checkpoint, (
        network, optimizer.init(eqx.filter(network, eqx.is_inexact_array))))
    tau_u, tau_v = swg_surface_wind(small_card.resolved_namelist)
    forcing = OceanSurfaceForcing(tau_i_native=jnp.asarray(tau_u),
                                  tau_j_native=jnp.asarray(tau_v))
    state = driver._double_gyre_driver().seed_steady_forcing_carry(
        small_card.recipe.initial_state, forcing)
    noise = jnp.zeros((3,) + tuple(geometry.tmask.shape))
    step = jax.jit(lambda current: swg_online_step(
        network, current, noise, jax.random.PRNGKey(11), jnp.asarray(0.0),
        card=small_card, geometry=geometry, forcing=forcing))
    total = [0.0, 0.0, 0.0]
    for _ in range(24):
        state = step(state)
        total = [value + np.asarray(field) for value, field in zip(total, (
            state.eta.data, state.u.data[:, 1:, 0], state.v.data[1:, :, 0]))]
    for mean, value in zip(stored, total, strict=True):
        np.testing.assert_allclose(mean, value / 24, rtol=0.0,
                                   atol=1.0e-12 * np.max(np.abs(value / 24)))

    # (1) Corrupted manifests are refused.
    def corrupted(name, edit):
        target = tmp_path / name
        target.mkdir()
        payload = json.loads((root / "manifest.json").read_text())
        edit(payload)
        (target / "manifest.json").write_text(json.dumps(payload))
        return evaluation.admit_trained_run(
            target, driver._sha256(target / "manifest.json"),
            objective="template", window_days=1)

    def round7_block(payload):
        closure = payload["closure"]
        closure["checkpoint"] = {"path": closure.pop("checkpoint_path"),
                                 "sha256": closure.pop("checkpoint_sha256")}

    for name, edit, match in (
            ("round7_schema", lambda payload: payload.update(
                schema="legoesm-swg-online-free-run-v1"), "manifest schema"),
            ("other_digest", lambda payload: payload["closure"].update(
                checkpoint_sha256="0" * 64), "differs from checkpoint_sha256"),
            ("round7_block", round7_block, "exactly the keys")):
        with pytest.raises(ValueError, match=match):
            corrupted(name, edit)

    # (3) The checkpoint file missing, then changed by one bit.
    original = checkpoint.read_bytes()
    checkpoint.rename(tmp_path / "moved.eqx")
    with pytest.raises(ValueError, match="checkpoint file is missing"):
        evaluation.admit_trained_run(root, sha, objective="template",
                                     window_days=1)
    checkpoint.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="differs from checkpoint_sha256"):
        evaluation.admit_trained_run(root, sha, objective="template",
                                     window_days=1)


def test_free_run_command_takes_its_device_from_the_configuration(
        tmp_path, monkeypatch, capsys):
    """``free-run`` has no device flag: the configuration's CPU is selected,
    and an environment set up for another platform is refused."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config(tmp_path / "record")))
    argv = ["free-run", "--config", str(path), "--seed", "11", "--run-root",
            str(tmp_path / "runs"), "--window-days", "1"]
    with pytest.raises(SystemExit):
        driver.main(argv + ["--device", "cuda"])
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    assert driver.main(argv) == 2
    assert "JAX_PLATFORMS must be exactly 'cpu'" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_dg50_free_run_deck_passes_the_scorers_deck_binding():
    """The held-out scorer binds a free run to the dg50 deck by namelist."""
    namelist = driver.DECKS["dg50"]
    card = types.SimpleNamespace(resolved_namelist=namelist,
                                 dt_s=namelist.rn_Dt_s)
    resolved = driver.free_run_namelist(card, 7300)
    assert resolved == driver._double_gyre_driver().resolved_daily_namelist(
        7300, deck="dg50")
    evaluation.climate._require_legoesm_deck(
        json.loads(json.dumps(resolved._asdict())), namelist)
    with pytest.raises(Exception, match="deck differs at nn_write"):
        evaluation.climate._require_legoesm_deck(
            json.loads(json.dumps(namelist._replace(nn_write=1)._asdict())),
            namelist)


@pytest.mark.parametrize("control", ("nu10", "nu80"))
def test_cpu_control_command_reproduces_frozen_days_one_and_two_bitwise(
        tmp_path, monkeypatch, control):
    """The new control path is the frozen control path, not a new scheme."""
    from legoesm.ocean.fidelity import provenance

    config = json.loads(CONTROL_CONFIGS[control].read_text())
    config["experiment"] = f"test_{control}"
    config["free_run"].update(days=2, device="cpu", math_library="libm")
    path = _write_config(tmp_path / f"{control}.json", config)
    monkeypatch.setattr(provenance, "git_sha", lambda **kwargs: "0" * 40)
    assert driver.main([
        "free-run", "--config", str(path), "--run-root", str(tmp_path)]) == 0
    root = tmp_path / config["experiment"]
    frozen = evaluation.CONTROL_ROOTS[control]
    for day in (1, 2):
        with np.load(root / f"day{day:04d}.npz") as actual, \
                np.load(frozen / f"day{day:04d}.npz") as expected:
            for field in SNAPSHOT_FIELDS:
                assert np.array_equal(actual[field], expected[field]), \
                    (control, day, field,
                     np.max(np.abs(actual[field] - expected[field])))


def test_resume_refuses_a_changed_configuration_or_execution(
        trained, small_card):
    folder, config = trained["folder"], trained["config"]
    changed = copy.deepcopy(config)
    changed["optimiser"]["learning_rate"] = 2.0e-3
    path = _write_config(folder / "changed.json", changed)
    with pytest.raises(RuntimeError, match="differs in"):
        driver.train(changed, path, card=small_card, seed=11,
                     run_dir=folder / "run", device=CPU, windows_per_call=10,
                     commit="0" * 40)
    with pytest.raises(RuntimeError, match="execution"):
        driver.train(config, trained["path"], card=small_card, seed=11,
                     run_dir=folder / "run", device=CPU, windows_per_call=5,
                     commit="0" * 40)
    with pytest.raises(RuntimeError, match="legoesm_git_sha"):
        driver.train(config, trained["path"], card=small_card, seed=11,
                     run_dir=folder / "run", device=CPU, windows_per_call=10,
                     commit="1" * 40)
    with pytest.raises(ValueError, match="not one of"):
        driver.train(config, trained["path"], card=small_card, seed=12,
                     run_dir=folder / "c", device=CPU, windows_per_call=10,
                     commit="0" * 40)


_REAL_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                    "swgonl_rounds/round6/hourly_dual_12y")
_REAL_CHUNKS = tuple(f"dual_day{day:04d}_to_day{day + 30:04d}"
                     for day in range(731, 911, 30))


@pytest.mark.skipif(not _REAL_RECORD.is_dir(),
                    reason="round-6 hourly record not mounted")
def test_every_committed_record_label_names_complete_chunks_on_disk():
    """Round 8: the test-year chunks are ``dual_test_day...``; a configuration
    labelling them ``dual`` pins none and the contract refuses its test split."""
    for path in (*CONFIGS, PILOT):
        for record in driver.load_config(path)["dataset"]["records"]:
            assert driver.complete_chunks(record), (path.name, record)


@pytest.mark.skipif(
    not all((_REAL_RECORD / name / "manifest.json").is_file()
            for name in _REAL_CHUNKS),
    reason="round-6 hourly chunks for days 731-911 are not complete")
def test_real_chunks_build_the_contract_and_pass_the_start_gates(
        tmp_path, monkeypatch, capsys):
    """The real record's first six chunks through the driver's data path."""
    config = driver.load_config(
        _ROOT / "config/swg_online/exp1_template_ring_excluded.json")
    config["dataset"]["records"] = [
        {"root": str(_REAL_RECORD), "chunk_label": "dual"}]
    config["dataset"]["splits_days"] = {
        "train": [731, 761], "validation": [791, 821], "test": [851, 881]}
    payload = driver.write_dataset_contract(config, tmp_path / "contract.json")
    assert sorted(Path(source["root"]).name
                  for source in payload["sources"]) == list(_REAL_CHUNKS)
    card = build_swg_card(driver.DECKS["dg50"])
    contract = driver.SWGDatasetContract.from_json(
        tmp_path / "contract.json", card=card)
    geometry = build_closure_geometry(card)
    spec, scales = driver.network_spec(config, card, geometry)
    seed = config["seeds"][0]
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(seed))
    gates = driver._start_gates(network, contract, card, geometry, scales,
                                window_days=1, seed=seed,
                                key=jax.random.PRNGKey(seed))
    hours = np.asarray(gates["start_absolute_hours"])
    assert hours.size == 10 and len(set(hours)) == 10
    assert hours.min() >= 731 * 24 and hours.max() + 24 <= 761 * 24
    assert gates["initial_closure_acceleration"]["passes"]
    report = gates["eq6_scale_report"]
    # Dimensionless inputs of order one, as the template's own input
    # zeta/(|beta| L) is on the same states; a small initial output.
    for name in ("ssh", "u_t", "v_t", "template_input_zeta"):
        assert 0.1 < report[name]["max_abs"] < 10.0, name
    assert report["output_potential"]["max_abs"] < 1.0
    # The command line: every split at windows of one and two days.
    config["curriculum"]["window_days"] = [1, 2]
    path = _write_config(tmp_path / "config.json", config)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    previous = get_policy()
    try:
        assert driver.main(["check-data", "--config", str(path), "--device",
                            "cpu", "--output", str(tmp_path / "check.json")]) \
            == 0, capsys.readouterr().err
    finally:
        set_policy(previous)
    checked = json.loads((tmp_path / "check.json").read_text())
    assert len(checked["batches"]) == 6
    assert checked["contract_sha256"] == driver._sha256(
        tmp_path / "contract.json")
    assert "DATA_CHECKED" in capsys.readouterr().out


def test_profile_times_a_ten_window_step_and_extrapolates_one_seed(
        small_card, record):
    config = _config(record)
    config["curriculum"].update(window_days=[1], epochs_per_window=5)
    report = driver.profile(config, card=small_card, device=CPU,
                            windows_per_call=10)
    (row,) = report["windows"]
    assert row["window_days"] == 1 and row["loss_finite"]
    assert min(row["step_s"], row["forward_s"]) > 0.0
    # 22 training days and 20 validation days of one-day windows.
    assert (row["train_batches_per_epoch"],
            row["validation_batches_per_epoch"]) == (2, 2)
    assert row["epoch_s"] == pytest.approx(
        2 * row["step_s"] + 2 * row["forward_s"], rel=1.0e-12)
    assert row["window_s"] == pytest.approx(5 * row["epoch_s"], rel=1.0e-12)
    free = report["free_run"]
    assert free["seconds_per_day"] > 0.0
    assert free["seconds_per_run"] == pytest.approx(
        config["free_run"]["days"] * free["seconds_per_day"], rel=1.0e-12)
    estimate = report["one_seed_estimate"]
    assert estimate["total_hours"] == pytest.approx(
        (row["window_s"] + free["seconds_per_run"]) / 3600.0, rel=1.0e-12)


# ------------------------------------------------- record-state free-run start

def _record_start(record: Path, day: int, **changes) -> dict:
    return {"kind": "record_state",
            "record": {"root": str(record), "chunk_label": "dual"},
            "day": day, "objective": "template"} | changes


@pytest.mark.parametrize("edit,match", [
    (lambda s: s.update(kind="restart"), "record_state"),
    (lambda s: s.pop("day"), "exactly the keys"),
    (lambda s: s.update(day=-1), "integer >= 0"),
    (lambda s: s.update(objective="box"), "objective"),
    (lambda s: s["record"].pop("chunk_label"), "exactly the keys"),
    (lambda s: s.update(objective="symmetric"), "training objective"),
])
def test_config_refuses_a_malformed_or_mismatched_record_start(
        tmp_path, edit, match):
    config = _config(tmp_path / "record")
    start = _record_start(tmp_path / "record", 52)
    edit(start)
    config["free_run"]["start"] = start
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=match):
        driver.load_config(path)
    with pytest.raises(ValueError, match="rest"):
        driver._validate_free_run(dict(config["free_run"], start="truth"))


def test_record_start_is_the_datasets_window_start_bit_for_bit(
        small_card, record, tmp_path):
    """A free run started at day D resumes exactly the state the contract
    issues as the first validation window's start for that day: the same
    stored state, the same rebalancing, bit for bit."""
    config = _config(record, seeds=[11])
    config["free_run"]["start"] = _record_start(record, 52)
    path = _write_config(tmp_path / "config.json", config)
    contract_path = tmp_path / "contract.json"
    driver.write_dataset_contract(config, contract_path)
    contract = driver.SWGDatasetContract.from_json(contract_path, card=small_card)
    batch = next(contract.batches(split="validation", window_days=1))
    assert int(batch.start_absolute_hour[0]) == 52 * 24
    state, block = driver.resolve_start(driver.load_config(path), small_card)
    for index, field in enumerate((state.eta.data, state.u.data[:, 1:, 0],
                                   state.v.data[1:, :, 0])):
        assert np.array_equal(np.asarray(field), batch.initial[index][0])
    (ssh, u, v), report = contract.start_state(52 * 24)
    assert np.array_equal(np.asarray(state.eta.data), ssh)
    assert block["rebalance"]["report"] == report
    assert block["absolute_hour"] == 52 * 24 and block["day"] == 52
    assert block["chunk_root"].endswith("dual_day0052_to_day0072")
    assert state.tau_x_prev is None   # free_run seeds the steady wind carry
    # A day no complete chunk serves is refused, not substituted.
    config["free_run"]["start"] = _record_start(record, 40)
    with pytest.raises(ValueError, match="source gap"):
        driver.resolve_start(config, small_card)
    config["free_run"]["start"] = _record_start(tmp_path / "nowhere", 52)
    with pytest.raises((ValueError, OSError)):
        driver.resolve_start(config, small_card)


def test_free_run_from_a_record_start_records_it_and_differs_from_rest(
        trained, small_card, record, tmp_path):
    """The free-run block is the free run's own: a configuration equal to the
    training one except there runs the trained closure from the record
    start into its own folder, the manifest records the start and the
    rebalancing report, and the first day differs from the from-rest run."""
    config = copy.deepcopy(trained["config"])
    config["free_run"].update(days=1, start=_record_start(record, 52))
    path = _write_config(tmp_path / "config.json", config)
    run_dir = trained["folder"] / "run"
    manifest = driver.run_free_run(
        config, path, card=small_card, seed=11, run_dir=run_dir,
        window_days=1, device=CPU, commit="0" * 40,
        output=tmp_path / "from_day52")
    root = tmp_path / "from_day52"
    assert (root / "manifest.json").is_file()
    assert manifest["start"]["kind"] == "record_state"
    assert manifest["start"]["day"] == 52
    assert manifest["start"]["rebalance"]["report"]["iterations"] >= 0
    assert manifest["closure"]["config_sha256"] == json.loads(
        (run_dir / "run_manifest.json").read_text())["config_sha256"]
    assert driver.free_run_output(config, run_dir, 1, None) \
        == run_dir / "free_runs" / "w01_day0052"
    rest = copy.deepcopy(config)
    rest["free_run"]["start"] = "rest"
    driver.run_free_run(
        rest, _write_config(tmp_path / "rest.json", rest), card=small_card,
        seed=11, run_dir=run_dir, window_days=1, device=CPU, commit="0" * 40,
        output=tmp_path / "from_rest")
    with np.load(root / "day0001.npz") as started, \
            np.load(tmp_path / "from_rest/day0001.npz") as from_rest:
        assert not np.array_equal(started["u_m_s"], from_rest["u_m_s"])
    # A training key that differs is still refused.
    changed = copy.deepcopy(config)
    changed["optimiser"]["learning_rate"] = 2.0e-3
    with pytest.raises(ValueError, match="not this configuration"):
        driver.run_free_run(
            changed, _write_config(tmp_path / "changed.json", changed),
            card=small_card, seed=11, run_dir=run_dir, window_days=1,
            device=CPU, commit="0" * 40, output=tmp_path / "refused")


def test_committed_day4265_configs_select_the_record_start():
    pilot = driver.load_config(PILOT)
    diagnosis = driver.load_config(
        _ROOT / "config/swg_online/pilot_template_ring_excluded_day4265.json")
    assert diagnosis | {"free_run": pilot["free_run"],
                        "description": pilot["description"]} == pilot
    control = driver.load_config(
        _ROOT / "config/swg_online/control_nu10_gpu_day4265.json")
    for config in (diagnosis, control):
        assert config["free_run"]["days"] == 1095
        assert config["free_run"]["start"] == {
            "kind": "record_state", "day": 4265, "objective": "template",
            "record": {"root": str(_REAL_RECORD), "chunk_label": "dual"}}
    assert pilot["dataset"]["splits_days"]["validation"][0] == 4265
    assert control["control"] == "nu10"


# ------------------------------------------------------ curriculum extension

WEIGHTED_PILOT = _ROOT / "config/swg_online/pilot_template_ring_weighted.json"


def test_committed_w4_configs_extend_the_weighted_pilot():
    """The w=4 configuration is the weighted pilot with the four-day window
    appended (same experiment name, so it extends the pilot's run); its
    day-4265 twin differs from it only as the pilot's twin does from the
    pilot (the free-run block and the prose)."""
    pilot = driver.load_config(WEIGHTED_PILOT)
    w4 = driver.load_config(WEIGHTED_PILOT.with_name(
        "pilot_template_ring_weighted_w4.json"))
    twin = driver.load_config(WEIGHTED_PILOT.with_name(
        "pilot_template_ring_weighted_w4_day4265_20y.json"))
    pilot_twin = driver.load_config(WEIGHTED_PILOT.with_name(
        "pilot_template_ring_weighted_day4265_20y.json"))
    assert pilot["curriculum"]["window_days"] == [1, 2]
    assert w4["curriculum"]["window_days"] == [1, 2, 4]
    assert w4 | {"description": pilot["description"],
                 "curriculum": pilot["curriculum"]} == pilot
    assert w4["curriculum"] | {"window_days": [1, 2]} == pilot["curriculum"]
    assert w4["experiment"] == pilot["experiment"]
    assert w4["description"] != pilot["description"]
    assert twin | {"description": w4["description"],
                   "free_run": w4["free_run"]} == w4
    assert twin["free_run"] == pilot_twin["free_run"]
    assert twin["free_run"]["days"] == 7300
    assert twin["free_run"]["start"]["day"] == 4265


@pytest.fixture(scope="module")
def one_window(tmp_path_factory, small_card, record):
    """A completed run of the one-day window alone (two epochs)."""
    folder = tmp_path_factory.mktemp("swg_extend")
    config = _config(record)
    config["curriculum"]["window_days"] = [1]
    path = _write_config(folder / "config.json", config)
    log = driver.train(config, path, card=small_card, seed=11,
                       run_dir=folder / "run", device=CPU,
                       windows_per_call=10, commit="0" * 40)
    assert log["status"] == "complete"
    return {"folder": folder, "config": config, "path": path}


def _extension(one_window, tmp_path, window_days=(1, 2)):
    """A scratch copy of the completed run and a configuration appending
    the two-day window to it (a new description, as the committed w=4 file
    has)."""
    run = tmp_path / "run"
    shutil.copytree(one_window["folder"] / "run", run)
    config = copy.deepcopy(one_window["config"])
    config["curriculum"]["window_days"] = list(window_days)
    config["description"] = "the one-day run continued to two days"
    return run, config


def _digests(run: Path) -> dict:
    return {str(path.relative_to(run)): driver._sha256(path)
            for path in sorted(run.rglob("*")) if path.is_file()}


def test_extension_appends_a_window_and_keeps_the_finished_stages(
        one_window, tmp_path, small_card):
    run, config = _extension(one_window, tmp_path)
    path = _write_config(tmp_path / "extended.json", config)
    before = _digests(run)
    stored = json.loads((run / "run_manifest.json").read_text())
    finished = json.loads((run / "training_log.json").read_text())["epochs"]
    assert [(e["window_days"], e["epoch"]) for e in finished] == [(1, 1), (1, 2)]

    log = driver.train(config, path, card=small_card, seed=11, run_dir=run,
                       device=CPU, windows_per_call=10, commit="2" * 40)

    # The finished stages: checkpoints bit-identical, log entries unchanged.
    after = _digests(run)
    assert {name: after[name] for name in before if "checkpoints/" in name} \
        == {name: before[name] for name in before if "checkpoints/" in name}
    assert log["epochs"][:2] == finished
    assert [(e["window_days"], e["epoch"]) for e in log["epochs"]] \
        == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert log["status"] == "complete"
    # The new stage: optimiser reinitialised, network carried from the
    # last finished checkpoint (the existing between-windows test's bars).
    steps = {(e["window_days"], e["epoch"]): e["optimizer_steps"]
             for e in log["epochs"]}
    end_one, _, initial = _params(
        run / "checkpoints/w01_e02.eqx", small_card, config, 11)
    start_two, count_two, _ = _params(
        run / "checkpoints/w02_e01.eqx", small_card, config, 11)
    assert count_two and set(count_two) == {steps[(2, 1)]}
    lr = config["optimiser"]["learning_rate"]
    assert all(np.max(np.abs(new - old)) <= 1.01 * lr
               for new, old in zip(start_two, end_one, strict=True))
    assert any(np.max(np.abs(new - old)) > 1.5 * lr
               for new, old in zip(start_two, initial, strict=True))
    # The record: manifest now the extending configuration's, with the
    # extension (from what, to what, at which commit) in it and in the log.
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["config"] == config
    assert manifest["config_sha256"] == driver._sha256(path)
    assert manifest["curriculum"]["window_days"] == [1, 2]
    assert manifest["legoesm_git_sha"] == "2" * 40
    (extension,) = manifest["curriculum_extensions"]
    assert extension.pop("extended_utc")
    assert extension == {
        "from_window_days": [1], "to_window_days": [1, 2],
        "epochs_per_window": 2, "completed_epochs_at_extension": 2,
        "from_config_sha256": stored["config_sha256"],
        "to_config_sha256": driver._sha256(path),
        "from_legoesm_git_sha": "0" * 40, "legoesm_git_sha": "2" * 40}
    assert log["invocations"][-1]["completed_epochs_at_start"] == 2
    assert log["invocations"][-1]["curriculum_extension"] \
        | {"extended_utc": None} == extension | {"extended_utc": None}
    assert driver.run_identity(manifest) | {
        key: stored[key] for key in driver._EXTENSION_KEYS} == stored

    # The extended run resumes as any other: identical, nothing to do; the
    # commit it now records is the extending one.
    again = driver.train(config, path, card=small_card, seed=11, run_dir=run,
                         device=CPU, windows_per_call=10, commit="2" * 40)
    assert again["epochs"] == log["epochs"]
    with pytest.raises(RuntimeError, match="legoesm_git_sha"):
        driver.train(config, path, card=small_card, seed=11, run_dir=run,
                     device=CPU, windows_per_call=10, commit="0" * 40)
    # Free runs bind to the extended configuration, not the one it extends.
    with pytest.raises(ValueError, match="not this configuration"):
        driver.run_free_run(
            one_window["config"], one_window["path"], card=small_card,
            seed=11, run_dir=run, window_days=1, device=CPU, commit="2" * 40)
    free = driver.run_free_run(
        config, path, card=small_card, seed=11, run_dir=run, window_days=2,
        device=CPU, commit="2" * 40)
    assert free["closure"]["window_days"] == 2
    assert free["closure"]["checkpoint_sha256"] \
        == log["epochs"][-1]["checkpoint"]["sha256"]
    assert free["closure"]["config_sha256"] == driver._sha256(path)


def test_extension_refuses_anything_but_appended_windows(
        one_window, tmp_path, small_card):
    def refused(config, path, match, run):
        untouched = _digests(run)
        with pytest.raises(RuntimeError, match=match):
            driver.train(config, path, card=small_card, seed=11, run_dir=run,
                         device=CPU, windows_per_call=10, commit="2" * 40)
        assert _digests(run) == untouched

    # Anything else changed beside the appended window: refused.
    run, config = _extension(one_window, tmp_path / "a")
    config["optimiser"]["learning_rate"] = 2.0e-3
    refused(config, _write_config(tmp_path / "a.json", config), "differs in",
            run)
    run, config = _extension(one_window, tmp_path / "b")
    config["curriculum"]["epochs_per_window"] = 3
    refused(config, _write_config(tmp_path / "b.json", config), "differs in",
            run)
    # The same windows under a new description: an ordinary changed
    # configuration.  A removed window, a replaced window or a reordered
    # list: not an extension.
    for name, windows, match in (("c", [1], "differs in"),
                                 ("d", [2], "do not extend"),
                                 ("e", [2, 1], "do not extend"),
                                 ("g", [2, 1, 4], "do not extend")):
        run, config = _extension(one_window, tmp_path / name, windows)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(config))   # [2, 1] never loads; direct
        refused(config, path, match, run)
    # A run whose own curriculum has not completed: resume it first.
    run, config = _extension(one_window, tmp_path / "f")
    log_path = run / "training_log.json"
    log = json.loads(log_path.read_text())
    log["epochs"] = log["epochs"][:1]
    log["status"] = "running"
    log_path.write_text(json.dumps(log))
    refused(config, _write_config(tmp_path / "f.json", config),
            "has not completed", run)


# ------------------------------------------------ pretrained initialisation

def _pretrained_file(path: Path, card, config, seed, scale):
    """A checkpoint in the driver's format holding ``seed``'s initialisation
    with every weight scaled by ``scale``; returns its network leaves."""
    geometry = build_closure_geometry(card)
    spec, _ = driver.network_spec(config, card, geometry)
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(seed))
    params, static = eqx.partition(network, eqx.is_inexact_array)
    network = eqx.combine(jax.tree.map(lambda leaf: scale * leaf, params), static)
    optimizer = create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))
    driver.save_checkpoint(
        (network, optimizer.init(eqx.filter(network, eqx.is_inexact_array))),
        path)
    return [np.asarray(leaf) for leaf in jax.tree.leaves(
        eqx.filter(network, eqx.is_inexact_array))]


def _one_epoch(record, **changes):
    config = _config(record, **changes)
    config["curriculum"].update(window_days=[1], epochs_per_window=1)
    return config


def _train(config, path, card, run_dir):
    return driver.train(config, path, card=card, seed=11, run_dir=run_dir,
                        device=CPU, windows_per_call=10, commit="0" * 40)


def test_initial_checkpoint_loads_the_file_and_records_its_digest(
        tmp_path, small_card, record):
    file = tmp_path / "pretrained.eqx"
    config = _one_epoch(record)
    expected = _pretrained_file(file, small_card, config, 11, 0.5)
    config["network"]["initial_checkpoint"] = str(file)
    path = _write_config(tmp_path / "config.json", config)
    seen = []
    gates = driver._start_gates

    def capturing(network, *args, **kwargs):
        seen.append([np.asarray(leaf) for leaf in jax.tree.leaves(
            eqx.filter(network, eqx.is_inexact_array))])
        return gates(network, *args, **kwargs)

    driver._start_gates = capturing
    try:
        log = _train(config, path, small_card, tmp_path / "run")
    finally:
        driver._start_gates = gates
    assert log["status"] == "complete"
    # The network at the first step is the file's, leaf for leaf.
    assert all(np.array_equal(mine, theirs)
               for mine, theirs in zip(seen[0], expected, strict=True))
    digest = {"path": str(file), "sha256": driver._sha256(file)}
    manifest = json.loads((tmp_path / "run/run_manifest.json").read_text())
    assert manifest["network"]["initial_checkpoint"] == digest
    assert "initial_checkpoint" in manifest["seed"]["network_init"]
    assert log["initial_checkpoint"] == digest


def test_initial_checkpoint_refuses_a_missing_or_mismatching_file(
        tmp_path, small_card, record, capsys):
    config = _one_epoch(record)
    config["network"]["initial_checkpoint"] = str(tmp_path / "absent.eqx")
    path = _write_config(tmp_path / "config.json", config)
    assert driver.main(["train", "--config", str(path), "--device", "cpu",
                        "--windows-per-call", "10", "--seed", "11",
                        "--run-root", str(tmp_path / "root")]) == 2
    assert "REFUSE" in capsys.readouterr().err
    assert not (tmp_path / "root").exists()
    # A file of another architecture: refused before any run directory.
    other = _one_epoch(record)
    other["network"]["hidden_channels"] = 8
    _pretrained_file(tmp_path / "other.eqx", small_card, other, 11, 1.0)
    config["network"]["initial_checkpoint"] = str(tmp_path / "other.eqx")
    path = _write_config(tmp_path / "config.json", config)
    with pytest.raises(CheckpointStructureError):
        _train(config, path, small_card, tmp_path / "run")
    assert not (tmp_path / "run").exists()
    # A file changed after the run started: refused on resume.
    _pretrained_file(tmp_path / "other.eqx", small_card, config, 11, 0.5)
    log = _train(config, path, small_card, tmp_path / "run")
    assert log["status"] == "complete"
    _pretrained_file(tmp_path / "other.eqx", small_card, config, 11, 0.25)
    with pytest.raises(RuntimeError, match="differs in"):
        _train(config, path, small_card, tmp_path / "run")


def test_without_initial_checkpoint_the_seed_initialises_as_before(
        tmp_path, small_card, record):
    plain = _one_epoch(record)
    assert "initial_checkpoint" not in plain["network"]
    log = _train(plain, _write_config(tmp_path / "plain.json", plain),
                 small_card, tmp_path / "plain")
    manifest = json.loads((tmp_path / "plain/run_manifest.json").read_text())
    assert "initial_checkpoint" not in manifest["network"]
    assert "initial_checkpoint" not in log
    assert manifest["seed"]["network_init"] \
        == "SWGScalarPotentialCNN(spec, key=PRNGKey(seed))"
    # A file holding exactly the seed's initialisation reproduces the plain
    # run's first stage bit for bit: absent means the seed's initialisation.
    file = tmp_path / "seed_init.eqx"
    _pretrained_file(file, small_card, plain, 11, 1.0)
    seeded = copy.deepcopy(plain)
    seeded["network"]["initial_checkpoint"] = str(file)
    seeded_log = _train(seeded, _write_config(tmp_path / "seeded.json", seeded),
                        small_card, tmp_path / "seeded")
    assert seeded_log["epochs"][0]["train_loss_per_batch"] \
        == log["epochs"][0]["train_loss_per_batch"]
    assert seeded_log["epochs"][0]["checkpoint"]["sha256"] \
        == log["epochs"][0]["checkpoint"]["sha256"]
