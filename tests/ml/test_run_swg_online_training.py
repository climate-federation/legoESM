"""Direct tests for the SWG online training driver (experiment one)."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import types
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card
from legoesm.training.swg_online import (
    SWGOptimizerSettings,
    SWGScalarPotentialCNN,
    build_closure_geometry,
    build_segment_fn,
    coarse_state_from_native,
    create_swg_optimizer,
    deterministic_trajectory_loss,
    maddison_eq6_network_scales,
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
CONFIGS = sorted((_ROOT / "config/swg_online").glob("*.json"))
CPU = {"device": "cpu", "backend": "cpu", "precision_policy": "fp64",
       "transcendentals": "libm", "physical_gpu": None,
       "cuda_visible_devices": None, "jax_platforms": "cpu",
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
        assert config["free_run"] == {
            "days": 7300, "start": "rest", "sampling": "daily_snapshot"}
        assert config["deck"] == "dg50"


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
    (lambda c: c.update(seeds=[1, 2]), "three distinct"),
    (lambda c: c.update(seeds=[1, 1, 2]), "three distinct"),
    (lambda c: c["curriculum"].update(window_days=[1, 16]), "window_days"),
    (lambda c: c["curriculum"].update(window_days=[2, 1]), "window_days"),
    (lambda c: c["curriculum"].update(optimiser_between_windows="carried"),
     "reinitialised"),
    (lambda c: c["dataset"].update(batch_size=8), "batch_size"),
    (lambda c: c["network"].update(scales="unit"), "scales"),
    (lambda c: c["free_run"].update(start="restart"), "rest"),
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
        driver.select_device("cuda", 1, environ={
            "JAX_PLATFORMS": "gpu", "CUDA_VISIBLE_DEVICES": "1"})
    with pytest.raises(RuntimeError, match="exactly 'cuda'"):
        driver.select_device("cuda", 1, environ={"CUDA_VISIBLE_DEVICES": "1"})
    with pytest.raises(ValueError, match="--physical-gpu"):
        driver.select_device("cuda", None, environ={"JAX_PLATFORMS": "cuda"})
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        driver.select_device("cuda", 1, environ={
            "JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": "0"})
    with pytest.raises(ValueError, match="only valid"):
        driver.select_device("cpu", 0, environ={"JAX_PLATFORMS": "cpu"})
    with pytest.raises(RuntimeError, match="exactly 'cpu'"):
        driver.select_device("cpu", None, environ={})
    with pytest.raises(RuntimeError, match="selected backend"):
        driver.select_device("cuda", 1, environ={
            "JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": "1"})
    previous = get_policy()
    try:
        assert driver.select_device(
            "cpu", None, environ={"JAX_PLATFORMS": "cpu"})["transcendentals"] \
            == "libm"
        monkeypatch.setattr(driver.jax, "default_backend", lambda: "gpu")
        selected = driver.select_device("cuda", 1, environ={
            "JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": "1"})
        assert selected["transcendentals"] == "native"
        assert selected["physical_gpu"] == 1
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
        seeds=[11, 7, 4],
        free_run={"days": 1, "start": "rest", "sampling": "daily_snapshot"})
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


def test_free_runs_are_scored_by_the_harness_and_identify_the_closure(trained):
    log = trained["log"]
    assert [run["window_days"] for run in log["free_runs"]] == [1, 2]
    for run in log["free_runs"]:
        root = Path(run["root"])
        manifest = json.loads((root / "manifest.json").read_text())
        assert run["manifest_sha256"] == driver._sha256(root / "manifest.json")
        final = next(entry for entry in log["epochs"]
                     if entry["window_days"] == run["window_days"]
                     and entry["epoch"] == 2)
        closure = manifest["closure"]
        assert closure["checkpoint"]["sha256"] == final["checkpoint"]["sha256"]
        assert closure["network_seed"] == 11
        assert closure["window_days"] == run["window_days"]
        assert closure["objective"] == "template"
        assert closure["run_id"] == "test_small/seed11"
        assert manifest["sampling"] == "daily_snapshot"
        assert manifest["start"] == "rest"
        ssh, u, v = evaluation._load_native_window(root, 1, 1)
        assert ssh.shape[0] == 1 and np.all(np.isfinite(ssh))
        assert np.any(u != 0.0)
        with np.load(root / "day0001.npz") as saved:
            assert str(saved["snapshot_sampling"]) \
                == evaluation.FINE_SNAPSHOT_LABEL
    assert log["free_runs"][0]["checkpoint_sha256"] \
        != log["free_runs"][1]["checkpoint_sha256"]


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
