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
from typing import NamedTuple

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
from legoesm.training import swg_online
from legoesm.training.swg_online import (
    DAILY_MEAN_FIELDS,
    SECONDS_PER_HOUR,
    SNAPSHOT_FIELDS,
    SWGOptimizerSettings,
    SWGScalarPotentialCNN,
    assert_network_gradients_live,
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
CONFIGS = sorted(path for path in (_ROOT / "config/swg_online").glob("exp1_*.json")
                 if "_day4265" not in path.name)
EXP1_WEIGHTED = _ROOT / "config/swg_online/exp1_template_ring_weighted.json"
EXP2_WALL_MARGIN = _ROOT / "config/swg_online/exp2_wallmargin2_template_ring_weighted.json"
EXP2_BIGNET = _ROOT / "config/swg_online/exp2_bignet_h6k5_template_ring_weighted.json"
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
        assert config["training_precision"] == "fp64"
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
            "wall_margin_rings": 0, "scales": driver.SCALE_RULE,
            "noise_channel": "none", "initial_acceleration_gate": "enforced"}
        assert config["loss"] == {"kind": "eq7"}
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


def test_exp2_arm_configs_differ_from_experiment_one_only_in_their_arm():
    base = driver.load_config(EXP1_WEIGHTED)
    wall = driver.load_config(EXP2_WALL_MARGIN)
    big = driver.load_config(EXP2_BIGNET)
    assert wall["experiment"] == "swgonl_exp2_wallmargin2_template_ring_weighted"
    assert big["experiment"] == "swgonl_exp2_bignet_h6k5_template_ring_weighted"
    for config, arm in ((wall, "Arm A"), (big, "Arm B")):
        assert config["description"].startswith(
            f"{arm} of the 2026-10-16 decision (user chose both arms): ")
        assert config["description"].endswith(base["description"])
    assert wall == base | {
        "experiment": wall["experiment"], "description": wall["description"],
        "network": base["network"] | {"wall_margin_rings": 2}}
    assert big == base | {
        "experiment": big["experiment"], "description": big["description"],
        "network": base["network"] | {"hidden_layers": 6, "kernel_size": 5}}
    assert wall["network"]["wall_margin_rings"] == 2
    assert big["network"]["wall_margin_rings"] == 0


def test_load_config_refuses_a_missing_or_negative_wall_margin(tmp_path):
    config = json.loads(EXP1_WEIGHTED.read_text())
    path = tmp_path / "config.json"
    del config["network"]["wall_margin_rings"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="wall_margin_rings"):
        driver.load_config(path)
    config["network"]["wall_margin_rings"] = -1
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="wall_margin_rings must be an integer >= 0"):
        driver.load_config(path)


def test_wall_margin_reaches_the_spec_and_the_run_manifest(small_card, one_window):
    config = driver.load_config(EXP2_WALL_MARGIN)
    spec, _ = driver.network_spec(
        config, small_card, build_closure_geometry(small_card))
    assert spec.wall_margin_rings == 2
    manifest = json.loads(
        (one_window["folder"] / "run/run_manifest.json").read_text())
    assert manifest["config"]["network"]["wall_margin_rings"] == 0
    assert manifest["network"]["spec"]["wall_margin_rings"] == 0


def test_free_run_refuses_a_changed_wall_margin(one_window, small_card):
    config = copy.deepcopy(one_window["config"])
    config["network"]["wall_margin_rings"] = 1
    with pytest.raises(ValueError, match="not this configuration"):
        driver.run_free_run(
            config, one_window["path"], card=small_card, seed=11,
            run_dir=one_window["folder"] / "run", window_days=1, device=CPU,
            commit="0" * 40)


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
            "surface": "explicit",
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


def test_rigid_lid_gpu_control_configs_name_the_lid_controls_exactly():
    for name, deck in (("nu10", "dg50-rl"), ("nu80", "dg50-nu80-rl")):
        config = driver.load_config(
            _ROOT / f"config/swg_online/control_{name}_rl_gpu.json")
        assert config == {
            "schema": driver.CONTROL_CONFIG_SCHEMA,
            "experiment": f"swgonl_control_{name}_rl_gpu",
            "description": config["description"],
            "control": name, "deck": deck, "surface": "rigid_lid",
            "closure": "none",
            "free_run": {
                "days": 7300, "start": "rest",
                "sampling": "end_of_day_snapshot_and_daily_mean",
                "device": "cuda", "math_library": "native"},
        }
        # Only the viscosity differs from the lid clock; a control never mixes
        # its surface with the other surface's deck.
        assert driver.DECKS[deck].rn_Dt_s == 10_800.0
        assert driver.DECKS[deck].rn_Lv_m == (10.0 if name == "nu10" else 80.0)
        for edit in ({"surface": "explicit"},
                     {"deck": "dg50" if name == "nu10" else "dg50-nu80"}):
            with pytest.raises(ValueError):
                _refuse(config | edit)


def _refuse(config):
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "control.json"
        path.write_text(json.dumps(config))
        driver.load_config(path)


@pytest.mark.parametrize("edit,match", [
    (lambda c: c.pop("objective"), "exactly the keys"),
    (lambda c: c.pop("wall_ring"), "exactly the keys"),
    (lambda c: c.update(extra=1), "exactly the keys"),
    (lambda c: c.update(objective="auto"), "objective"),
    (lambda c: c.update(wall_ring=None), "wall_ring"),
    (lambda c: c.pop("training_precision"), "exactly the keys"),
    (lambda c: c.update(training_precision="fp16"), "training_precision"),
    (lambda c: c.update(training_precision="FP32"), "training_precision"),
    (lambda c: c.update(training_precision=32), "training_precision"),
    (lambda c: c.pop("gradient_checkpointing"), "exactly the keys"),
    (lambda c: c.update(gradient_checkpointing="yes"), "gradient_checkpointing"),
    (lambda c: c.update(gradient_checkpointing=1), "gradient_checkpointing"),
    (lambda c: c["dataset"].pop("cache"), "exactly the keys"),
    (lambda c: c["dataset"].update(cache="disk"), "cache"),
    (lambda c: c["dataset"].update(cache=True), "cache"),
    (lambda c: c["dataset"].pop("target_filter_sigma_cells"),
     "exactly the keys"),
    (lambda c: c["dataset"].update(target_filter_sigma_cells=0.5),
     "target_filter_sigma_cells"),
    (lambda c: c["dataset"].update(target_filter_sigma_cells=True),
     "target_filter_sigma_cells"),
    (lambda c: c["dataset"].update(target_filter_sigma_cells="2"),
     "target_filter_sigma_cells"),
    (lambda c: c["dataset"].update(target_filter_sigma_cells=float("nan")),
     "target_filter_sigma_cells"),
    # CONFIGS[0] is the symmetric candidate: it may not widen the filter.
    (lambda c: c["dataset"].update(target_filter_sigma_cells=1.5),
     "template objective only"),
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
    (lambda c: c.update(surface="free"), "surface"),
    (lambda c: c.pop("surface"), "exactly the keys"),
    (lambda c: c.pop("target_interval_hours"), "exactly the keys"),
    (lambda c: c.update(target_interval_hours=0), "target_interval_hours"),
    (lambda c: c.update(target_interval_hours=5), "target_interval_hours"),
    (lambda c: c.update(target_interval_hours="3"), "target_interval_hours"),
    (lambda c: c.update(target_interval_hours=True), "target_interval_hours"),
    (lambda c: c.update(target_interval_hours=2.0), "target_interval_hours"),
    # dg50 steps at 400 s: 3 h is 27 steps, 8 h is not a divisor of a day.
    (lambda c: c.update(target_interval_hours=16), "target_interval_hours"),
    (lambda c: c.update(deck="dg50-rl", surface="rigid_lid"), "must divide"),
])
def test_config_refuses_missing_extra_and_unknown_choices(tmp_path, edit, match):
    config = json.loads(CONFIGS[0].read_text())
    edit(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=match):
        driver.load_config(path)


def test_rigid_lid_training_deck_requires_the_lid(tmp_path):
    config = json.loads(CONFIGS[0].read_text())
    config.update(deck="dg50-rl", surface="rigid_lid", target_interval_hours=3)
    path = tmp_path / "lid.json"
    path.write_text(json.dumps(config))
    assert driver.load_config(path)["deck"] == "dg50-rl"
    card = driver.build_deck_card("dg50-rl", "rigid_lid")
    assert card.dt_s == 10_800.0
    assert card.recipe.model_config.swe_surface == "rigid_lid"
    config["surface"] = "explicit"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="requires surface='rigid_lid'"):
        driver.load_config(path)
    with pytest.raises(ValueError, match="requires surface='rigid_lid'"):
        driver.build_deck_card("dg50-rl", "explicit")


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
                                  seed=11, target_interval_hours=1))
    geometry = build_closure_geometry(small_card)
    spec, _ = driver.network_spec(config, small_card, geometry)
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(11))
    _, segment = build_segment_fn(small_card, geometry, window_days=1, target_interval_hours=1)
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

    def planted(value_and_grad, network, batch, key, windows_per_call,
                **_):
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
                                key=jax.random.PRNGKey(seed),
                                target_interval_hours=1)
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
    batch = next(contract.batches(split="validation", window_days=1, target_interval_hours=1))
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
    run, config = _extension(one_window, tmp_path / "h")
    config["network"]["wall_margin_rings"] = 1
    refused(config, _write_config(tmp_path / "h.json", config), "differs in",
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


# ------------------------------------------------------- training precision

EXP3_FP32 = _ROOT / "config/swg_online/exp3_symmetric_ring_weighted_fp32.json"
EXP1_SYMMETRIC = _ROOT / "config/swg_online/exp1_symmetric_ring_weighted.json"


def test_every_committed_training_config_states_its_training_precision():
    trained_configs = [
        path for path in sorted((_ROOT / "config/swg_online").glob("*.json"))
        if "network" in json.loads(path.read_text())
        and path.name not in PENDING_CONFIGS]
    assert len(trained_configs) >= 17
    precisions = {path.name: driver.load_config(path)["training_precision"]
                  for path in trained_configs}
    assert precisions.pop(EXP3_FP32.name) == "fp32"
    assert precisions.pop(EXP4_TEMPLATE.name) == "fp32"
    assert precisions.pop(EXP4_SYMMETRIC.name) == "fp32"
    assert precisions.pop(EXP5_TEMPLATE.name) == "fp32"
    assert precisions.pop(EXP5_SYMMETRIC.name) == "fp32"
    for path in EXP6:
        assert precisions.pop(path.name) == "fp32"
    assert precisions.pop(EXP4_SYMMETRIC_LONG.name) == "fp32"
    assert precisions.pop(EXP7.name) == "fp32"
    assert precisions.pop(EXP8.name) == "fp32"
    assert set(precisions.values()) == {"fp64"}


EXP5_SYMMETRIC = _ROOT / "config/swg_online/exp5_rigidlid_symmetric_fast.json"
EXP5_TEMPLATE = _ROOT / "config/swg_online/exp5_rigidlid_template_fast.json"
EXP6 = tuple(_ROOT / f"config/swg_online/exp6_rigidlid_template_{tag}.json"
             for tag in ("sigma1p5", "sigma2"))
EXP4_TEMPLATE = _ROOT / "config/swg_online/exp4_rigidlid_template_ring_weighted_fp32.json"
EXP4_SYMMETRIC = _ROOT / "config/swg_online/exp4_rigidlid_symmetric_ring_weighted_fp32.json"
EXP4_SYMMETRIC_LONG = EXP4_SYMMETRIC.with_name(
    "exp4_rigidlid_symmetric_ring_weighted_fp32_long.json")
RIGID_LID_TRUTH = ("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                   "swgonl_rounds/round16_rigidlid_truth/")


def test_every_committed_training_config_states_its_target_interval():
    trained_configs = [
        path for path in sorted((_ROOT / "config/swg_online").glob("*.json"))
        if "network" in json.loads(path.read_text())
        and path.name not in PENDING_CONFIGS]
    intervals = {path.name: driver.load_config(path)["target_interval_hours"]
                 for path in trained_configs}
    assert intervals.pop(EXP4_TEMPLATE.name) == 3
    assert intervals.pop(EXP4_SYMMETRIC.name) == 3
    assert intervals.pop(EXP5_TEMPLATE.name) == 3
    assert intervals.pop(EXP5_SYMMETRIC.name) == 3
    for path in EXP6:
        assert intervals.pop(path.name) == 3
    assert intervals.pop(EXP4_SYMMETRIC_LONG.name) == 3
    assert intervals.pop(EXP7.name) == 3
    assert intervals.pop(EXP8.name) == 3
    assert len(intervals) >= 17 and set(intervals.values()) == {1}


@pytest.mark.parametrize("parent,child,experiment", [
    (EXP1_WEIGHTED, EXP4_TEMPLATE,
     "swgonl_exp4_rigidlid_template_ring_weighted_fp32"),
    (EXP3_FP32, EXP4_SYMMETRIC,
     "swgonl_exp4_rigidlid_symmetric_ring_weighted_fp32")])
def test_exp4_rigid_lid_configs_differ_from_their_parent_only_as_decided(
        parent, child, experiment):
    base, new = driver.load_config(parent), driver.load_config(child)
    prefix = ("Rigid-lid training on the rigid-lid truth (user decisions "
              "2026-10-18: rigid lid, phi/g input, 3-hour step and 3-hour "
              "loss sampling, both objectives): ")
    assert new["description"].startswith(prefix + base["description"])
    assert "PLACEHOLDERS" in new["description"]
    records = [{"root": RIGID_LID_TRUTH + name, "chunk_label": label}
               for name, label in (("hourly_dual_12y", "dual"),
                                   ("hourly_dual_test_year", "dual_test"))]
    assert new == base | {
        "experiment": new["experiment"], "description": new["description"],
        "deck": "dg50-rl", "surface": "rigid_lid", "training_precision": "fp32",
        "target_interval_hours": 3,
        "dataset": base["dataset"] | {"records": records}}
    assert new["experiment"] == experiment
    assert new["free_run"]["device"] == "cuda" \
        and new["free_run"]["math_library"] == "native"


@pytest.mark.parametrize("config_path", [EXP4_TEMPLATE, EXP4_SYMMETRIC])
def test_exp4_refuses_to_load_data_until_the_rigid_lid_truth_exists(
        config_path, tmp_path):
    config = driver.load_config(config_path)
    for index, record in enumerate(config["dataset"]["records"]):
        record["root"] = str(tmp_path / f"missing_record_{index}")
        with pytest.raises(ValueError, match="record root does not exist"):
            driver.complete_chunks(record)
    card = driver.build_deck_card(config["deck"], config["surface"])
    with pytest.raises(ValueError, match="record root does not exist"):
        driver.check_data(config, card=card)


def test_exp3_is_the_symmetric_candidate_in_single_precision_on_the_gpu():
    exp3 = driver.load_config(EXP3_FP32)
    symmetric = driver.load_config(EXP1_SYMMETRIC)
    template = driver.load_config(EXP1_WEIGHTED)
    assert exp3["experiment"] == "swgonl_exp3_symmetric_ring_weighted_fp32"
    assert exp3["description"] == (
        "Next run after the 2026-10-17 decision (user: symmetric objective "
        "next, single-precision training, GPU free runs): "
        + symmetric["description"].replace(
            "One of three candidates; the owner has not chosen the objective, "
            "and the wall-ring treatment for this candidate is also open.",
            "Wall ring weighted, as in the full experiment (user decision "
            "2026-10-15)."))
    assert exp3 == symmetric | {
        "experiment": exp3["experiment"], "description": exp3["description"],
        "training_precision": "fp32",
        "free_run": symmetric["free_run"] | {
            "device": "cuda", "math_library": "native"}}
    assert exp3["network"]["wall_margin_rings"] == 0
    assert exp3 | {key: template[key] for key in (
        "experiment", "description", "objective", "training_precision")} \
        == template


def _parent_batch_functions(card, geometry, *, window_days, objective,
                            wall_ring):
    """``build_batch_functions`` as it was before ``training_precision``."""
    raw, _ = build_segment_fn(
        card, geometry, window_days=window_days, checkpoint=True, target_interval_hours=1)
    n_steps = (window_days * driver.HOURS_PER_DAY
               * int(SECONDS_PER_HOUR / card.dt_s))
    template = card.recipe.initial_state
    namelist = card.resolved_namelist
    shape = tuple(geometry.tmask.shape)

    def batch_loss(network, initial, target, key):
        noise = jnp.zeros((n_steps, 3) + shape, dtype=jnp.float64)

        def rollout(ssh, u, v):
            state = coarse_state_from_native(template, ssh, u, v)
            return raw(network, state, noise, key, jnp.asarray(0.0))[1]

        predicted = jax.vmap(rollout)(*initial)
        return deterministic_trajectory_loss(
            predicted, target, geometry, objective=objective,
            beta_m1_s1=namelist.rn_beta_m_s,
            basin_length_m=namelist.rn_domsiz_m, wall_ring=wall_ring)

    return (eqx.filter_jit(eqx.filter_value_and_grad(batch_loss)),
            eqx.filter_jit(batch_loss))


@pytest.fixture(scope="module")
def precision_steps(small_card):
    """One optimiser step of ten random one-day windows, three ways: the
    parent code path, ``"fp64"`` and ``"fp32"`` (built with a spy on the
    jitted functions that records their arguments and traced bodies)."""
    card = small_card
    geometry = build_closure_geometry(card)
    config = driver.load_config(EXP1_SYMMETRIC)
    spec, _ = driver.network_spec(config, card, geometry)
    key = jax.random.PRNGKey(11)
    network = SWGScalarPotentialCNN(spec, key=key)
    optimizer = create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))
    shape = tuple(geometry.tmask.shape)
    rng = np.random.default_rng(3)
    batch = driver.WindowBatch(
        tuple(1.0e-3 * rng.normal(size=(10,) + shape) for _ in range(3)),
        tuple(1.0e-3 * rng.normal(size=(10, 24) + shape) for _ in range(3)),
        np.arange(10), "train", 1)
    options = {"window_days": 1, "objective": config["objective"],
               "wall_ring": config["wall_ring"]}
    update = driver._update_function(optimizer)

    def step(functions):
        loss, grads = driver.batch_value_and_grad(
            functions[0], network, batch, key, 10)
        opt_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
        stepped, _ = update(network, grads, opt_state)
        return {"loss": loss, "grads": grads, "network": stepped}

    traced, calls = [], []
    real_filter_jit = eqx.filter_jit

    def spy(function):
        jitted = real_filter_jit(function)
        traced.append(function)

        def call(*args):
            calls.append(sorted({
                str(leaf.dtype) for leaf in jax.tree.leaves(
                    eqx.filter(args, eqx.is_inexact_array))}))
            return jitted(*args)
        return call

    eqx.filter_jit = spy
    try:
        functions32 = driver.build_batch_functions(
            card, geometry, training_precision="fp32", checkpoint=True, **options,
            target_interval_hours=1)
    finally:
        eqx.filter_jit = real_filter_jit
    return {
        "parent": step(_parent_batch_functions(card, geometry, **options)),
        "fp64": step(driver.build_batch_functions(
            card, geometry, training_precision="fp64", checkpoint=True, **options,
            target_interval_hours=1)),
        "fp32": step(functions32), "traced": traced[-1], "calls": calls,
        "network": network, "batch": batch, "key": key, "card": card,
        "geometry": geometry}


def test_fp64_training_precision_reproduces_the_parent_step_bit_for_bit(
        precision_steps):
    parent, fp64 = precision_steps["parent"], precision_steps["fp64"]
    assert float(fp64["loss"]) == float(parent["loss"]) > 0.0
    for name in ("grads", "network"):
        for new, old in zip(
                jax.tree.leaves(eqx.filter(fp64[name], eqx.is_inexact_array)),
                jax.tree.leaves(eqx.filter(parent[name], eqx.is_inexact_array)),
                strict=True):
            assert new.dtype == jnp.float64
            assert np.array_equal(np.asarray(new), np.asarray(old))


def test_fp32_step_keeps_float64_masters_and_matches_float64_closely(
        precision_steps):
    fp32, fp64 = precision_steps["fp32"], precision_steps["fp64"]
    assert fp32["loss"].dtype == jnp.float64
    for tree in (fp32["grads"], fp32["network"]):
        assert {leaf.dtype for leaf in jax.tree.leaves(
            eqx.filter(tree, eqx.is_inexact_array))} == {jnp.dtype("float64")}
    assert float(fp32["loss"]) != float(fp64["loss"])
    assert float(fp32["loss"]) == pytest.approx(float(fp64["loss"]), rel=1.0e-5)
    new = np.concatenate([np.ravel(leaf) for leaf in jax.tree.leaves(
        eqx.filter(fp32["grads"], eqx.is_inexact_array))])
    old = np.concatenate([np.ravel(leaf) for leaf in jax.tree.leaves(
        eqx.filter(fp64["grads"], eqx.is_inexact_array))])
    assert np.max(np.abs(new - old)) < 1.0e-4 * np.max(np.abs(old))


def _jaxpr_float64_use(jaxpr):
    """Walk ``jaxpr`` and every jaxpr nested in its equations (scan bodies,
    calls): ``(produced, carried, consumed, primitives)`` are the primitives
    with a float64 result, the dtypes of every scan's carry, the primitives
    without a nested jaxpr that read a float64 value, and all primitives."""
    produced, carried, consumed, primitives = set(), set(), set(), set()

    def is64(var):
        return str(var.aval.dtype) == "float64"

    def visit(inner):
        inner = getattr(inner, "jaxpr", inner)
        for equation in inner.eqns:
            name = equation.primitive.name
            nested = [item for param in equation.params.values()
                      for item in (param if isinstance(param, (list, tuple))
                                   else (param,))
                      if hasattr(getattr(item, "jaxpr", item), "eqns")]
            primitives.add(name)
            if any(is64(var) for var in equation.outvars):
                produced.add(name)
            if not nested and any(is64(var) for var in equation.invars
                                  if hasattr(var, "aval")):
                consumed.add(name)
            if name == "scan":
                first = equation.params["num_consts"]
                carried.update(
                    str(var.aval.dtype) for var in
                    equation.invars[first:first + equation.params["num_carry"]])
            for item in nested:
                visit(item)

    visit(jaxpr)
    return produced, carried, consumed, primitives


def test_fp32_trace_holds_loss_state_scan_and_network_in_float32(
        precision_steps):
    assert precision_steps["calls"] == [["float32"]]
    batch, traced = precision_steps["batch"], precision_steps["traced"]

    with driver._single_precision_trace():
        arguments = (driver._single(precision_steps["network"]),
                     driver._single(tuple(map(jnp.asarray, batch.initial))),
                     driver._single(tuple(map(jnp.asarray, batch.target))),
                     precision_steps["key"])
        loss = jax.make_jaxpr(traced)(*arguments)
        gradient = jax.make_jaxpr(eqx.filter_value_and_grad(traced))(*arguments)
    # The card's float64 constants (grid metrics, layer thicknesses) enter the
    # trace as closed-over values read only by a conversion to float32: no
    # operation produces a float64 value and every scan carries float32.
    for jaxpr in (loss, gradient):
        produced, carried, consumed, primitives = _jaxpr_float64_use(jaxpr)
        assert produced == set()
        assert carried == {"float32"}
        assert consumed <= {"convert_element_type"}
        assert {"scan", "dot_general"} <= primitives
    assert jax.config.jax_enable_x64


def test_free_runs_stay_float64_after_single_precision_training(
        precision_steps):
    card, geometry = precision_steps["card"], precision_steps["geometry"]
    assert jax.config.jax_enable_x64
    network = precision_steps["fp32"]["network"]
    forcing, one_day = driver.free_run_day(
        network, card, geometry, key=jax.random.PRNGKey(11))
    state = driver._double_gyre_driver().seed_steady_forcing_carry(
        card.recipe.initial_state, forcing)
    state, means = one_day(state)
    assert {str(array.dtype) for array in (
        state.eta.data, state.u.data, state.v.data, *means)} == {"float64"}
    assert {leaf.dtype for leaf in jax.tree.leaves(eqx.filter(
        network, eqx.is_inexact_array))} == {jnp.dtype("float64")}


def test_a_changed_training_precision_is_a_changed_run(one_window, small_card):
    manifest = json.loads(
        (one_window["folder"] / "run/run_manifest.json").read_text())
    assert manifest["config"]["training_precision"] == "fp64"
    config = copy.deepcopy(one_window["config"])
    config["training_precision"] = "fp32"
    path = _write_config(one_window["folder"] / "single.json", config)
    with pytest.raises(RuntimeError, match="differs in"):
        driver.train(config, path, card=small_card, seed=11,
                     run_dir=one_window["folder"] / "run", device=CPU,
                     windows_per_call=10, commit="0" * 40)
    with pytest.raises(ValueError, match="not this configuration"):
        driver.run_free_run(
            config, path, card=small_card, seed=11,
            run_dir=one_window["folder"] / "run", window_days=1, device=CPU,
            commit="0" * 40)


def test_build_batch_functions_refuses_an_unknown_training_precision(
        small_card):
    with pytest.raises(ValueError, match="training_precision"):
        driver.build_batch_functions(
            small_card, build_closure_geometry(small_card), window_days=1,
            objective="template", wall_ring="weighted",
            training_precision="bf16", checkpoint=True, target_interval_hours=1)


# ----------------------------------------- rigid-lid deck at a 3-hour step

@pytest.fixture(scope="module")
def lid_card():
    return driver.build_deck_card("dg50-rl", "rigid_lid")


def test_rigid_lid_deck_free_run_day_is_eight_steps_of_three_hours(lid_card):
    assert lid_card.dt_s == 10_800.0 and driver.steps_per_day(lid_card) == 8
    namelist = driver.free_run_namelist(lid_card, 2)
    assert (namelist.nn_itend, namelist.nn_write, namelist.nn_stock) == (16, 8, 8)
    geometry = build_closure_geometry(lid_card)
    key = jax.random.PRNGKey(0)
    forcing, one_day = driver.free_run_day(None, lid_card, geometry, key=key)
    start = driver._double_gyre_driver().seed_steady_forcing_carry(
        lid_card.recipe.initial_state, forcing)
    end, means = one_day(start)
    stage_noise = jnp.zeros((3,) + tuple(geometry.tmask.shape), jnp.float64)
    state = start
    for _ in range(8):
        state = driver.swg_online_step(
            None, state, stage_noise, key, jnp.asarray(0.0), card=lid_card,
            geometry=geometry, forcing=forcing)
    for new, old in ((end.u.data, state.u.data), (end.v.data, state.v.data)):
        np.testing.assert_allclose(np.asarray(new), np.asarray(old),
                                   rtol=1.0e-10, atol=1.0e-14)
    assert float(jnp.abs(end.u.data).max()) > 0.0       # the wind spun it up
    assert len(means) == 4     # ssh, u, v and the lid potential phi


@pytest.mark.parametrize("objective", ["symmetric", "template"])
def test_rigid_lid_three_hour_training_rollout_has_live_gradients(
        lid_card, objective):
    """Two one-day windows on the rigid-lid dg50 deck: the jitted batch loss
    and its gradient over 8 three-hourly records (phi/g closure input)."""
    config = driver.load_config(
        EXP4_SYMMETRIC if objective == "symmetric" else EXP4_TEMPLATE)
    assert config["objective"] == objective
    geometry = build_closure_geometry(lid_card)
    spec, _ = driver.network_spec(config, lid_card, geometry)
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(3))
    value_and_grad, loss_fn = driver.build_batch_functions(
        lid_card, geometry, window_days=1,
        target_interval_hours=config["target_interval_hours"],
        objective=objective, wall_ring=config["wall_ring"],
        training_precision="fp64", checkpoint=True)
    rng = np.random.default_rng(4)
    template = lid_card.recipe.initial_state
    u_shape = tuple(template.u.data[:, 1:, 0].shape)
    v_shape = tuple(template.v.data[1:, :, 0].shape)

    def native(*lead):
        return (np.zeros(lead + tuple(template.eta.data.shape)),
                1.0e-3 * rng.normal(size=lead + u_shape),
                1.0e-3 * rng.normal(size=lead + v_shape))

    initial, target = native(2), native(2, 8)
    key = jax.random.PRNGKey(1)
    loss, grads = value_and_grad(
        network, tuple(map(jnp.asarray, initial)),
        tuple(map(jnp.asarray, target)), key)
    assert np.isfinite(float(loss)) and float(loss) > 0.0
    assert_network_gradients_live(grads)       # finite and non-zero, every leaf
    assert len(jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))) == 9
    value = loss_fn(network, tuple(map(jnp.asarray, initial)),
                    tuple(map(jnp.asarray, target)), key)
    assert float(value) == pytest.approx(float(loss), rel=1.0e-10)
    # Hourly targets do not fit a 3-hourly rollout.
    wrong = native(2, 24)
    with pytest.raises(ValueError, match="share"):
        loss_fn(network, tuple(map(jnp.asarray, initial)),
                tuple(map(jnp.asarray, wrong)), key)


def test_rigid_lid_free_run_writes_whole_day_dual_files(tmp_path, lid_card):
    geometry = build_closure_geometry(lid_card)
    manifest = driver.free_run(
        None, lid_card, geometry, tmp_path / "free", days=2,
        closure={"kind": "none", "control": "nu10"}, commit="0" * 40,
        device=CPU, start=(None, "rest"))
    assert manifest["status"] == "complete" and manifest["steps_per_day"] == 8
    assert manifest["namelist"]["nn_write"] == 8
    for day in (1, 2):
        saved = np.load(tmp_path / "free" / f"day{day:04d}.npz")
        assert int(saved["step"]) == 8 * day and int(saved["day"]) == day
        assert float(saved["dt_s"]) == 10_800.0
        assert int(saved["daily_mean_sample_count"]) == 8
        assert set(driver.SNAPSHOT_FIELDS + driver.DAILY_MEAN_FIELDS) \
            <= set(saved.files)


# ------------------------------------------- gradient checkpointing and cache

def test_every_committed_training_config_states_checkpointing_and_cache():
    trained_configs = [
        path for path in sorted((_ROOT / "config/swg_online").glob("*.json"))
        if "network" in json.loads(path.read_text())
        and path.name not in PENDING_CONFIGS]
    states = {}
    for path in trained_configs:
        config = driver.load_config(path)
        states[path.name] = (config["gradient_checkpointing"],
                             config["dataset"]["cache"])
    # The arms keep checkpointing: 16 and 32 days have no measured no-checkpoint
    # peak (exp5 stopped at 8) and arm B holds 8 members per window.
    assert states.pop(EXP7.name) == (True, "memory")
    assert states.pop(EXP8.name) == (True, "memory")
    assert states.pop(EXP5_SYMMETRIC.name) == (False, "memory")
    assert states.pop(EXP5_TEMPLATE.name) == (False, "memory")
    for path in EXP6:
        assert states.pop(path.name) == (False, "memory")
    assert len(states) >= 17 and set(states.values()) == {(True, "none")}


@pytest.mark.parametrize("parent,child,experiment", [
    (EXP4_TEMPLATE, EXP5_TEMPLATE,
     "swgonl_exp5_rigidlid_template_fast"),
    (EXP4_SYMMETRIC, EXP5_SYMMETRIC,
     "swgonl_exp5_rigidlid_symmetric_fast")])
def test_exp5_fast_configs_differ_from_exp4_only_as_decided(
        parent, child, experiment):
    base, new = driver.load_config(parent), driver.load_config(child)
    assert new["description"] == (
        "Speed-up of exp4 (user 2026-10-18): in-memory record cache and no "
        "gradient checkpointing; numerically identical training. "
        + base["description"])
    assert new == base | {
        "experiment": experiment, "description": new["description"],
        "gradient_checkpointing": False,
        "dataset": base["dataset"] | {"cache": "memory"}}


_REMAT = {"checkpoint", "remat", "remat2"}


@pytest.fixture(scope="module")
def unrematerialised_step(precision_steps):
    """The ``precision_steps`` fp64 step again with both ``checkpoint``
    values, and the primitives of each build's loss and gradient programs."""
    card, geometry = precision_steps["card"], precision_steps["geometry"]
    batch, network = precision_steps["batch"], precision_steps["network"]
    key = precision_steps["key"]
    options = {"window_days": 1, "objective": "template",
               "wall_ring": "excluded", "training_precision": "fp64",
               "target_interval_hours": 1}
    arguments = (network, tuple(map(jnp.asarray, batch.initial)),
                 tuple(map(jnp.asarray, batch.target)), key)
    built, primitives = {}, {}
    for checkpoint in (True, False):
        value_and_grad, loss_fn = driver.build_batch_functions(
            card, geometry, checkpoint=checkpoint, **options)
        built[checkpoint] = driver.batch_value_and_grad(
            value_and_grad, network, batch, key, 10)
        primitives[checkpoint] = set().union(*(
            _jaxpr_float64_use(jax.make_jaxpr(function)(*arguments))[3]
            for function in (value_and_grad, loss_fn)))
    return built, primitives


def test_checkpoint_flag_changes_the_program_not_the_step(
        unrematerialised_step):
    built, primitives = unrematerialised_step
    assert _REMAT & primitives[True]            # the detector sees remat
    assert not _REMAT & primitives[False]       # and it is gone when off
    loss_on, grads_on = built[True]
    loss_off, grads_off = built[False]
    assert float(loss_on) > 0.0
    assert float(loss_off) == pytest.approx(float(loss_on), rel=1.0e-14)
    on = np.concatenate([np.ravel(leaf) for leaf in jax.tree.leaves(
        eqx.filter(grads_on, eqx.is_inexact_array))])
    off = np.concatenate([np.ravel(leaf) for leaf in jax.tree.leaves(
        eqx.filter(grads_off, eqx.is_inexact_array))])
    assert np.max(np.abs(on)) > 0.0
    assert np.max(np.abs(on - off)) <= 1.0e-14 * np.max(np.abs(on))


@pytest.mark.parametrize("checkpoint", [1, "false", None])
def test_build_batch_functions_refuses_a_non_boolean_checkpoint(
        small_card, checkpoint):
    with pytest.raises(ValueError, match="checkpoint"):
        driver.build_batch_functions(
            small_card, build_closure_geometry(small_card), window_days=1,
            objective="template", wall_ring="weighted",
            training_precision="fp64", checkpoint=checkpoint,
            target_interval_hours=1)


def test_memory_cache_training_reproduces_the_file_reading_run_bit_for_bit(
        one_window, small_card, tmp_path, capsys):
    config = copy.deepcopy(one_window["config"])
    assert config["dataset"]["cache"] == "none"
    config["dataset"]["cache"] = "memory"
    path = _write_config(tmp_path / "config.json", config)
    log = driver.train(config, path, card=small_card, seed=11,
                       run_dir=tmp_path / "run", device=CPU,
                       windows_per_call=10, commit="0" * 40)
    assert "DATASET_CACHE hours=" in capsys.readouterr().out
    assert log["status"] == "complete"
    reference = json.loads(
        (one_window["folder"] / "run/training_log.json").read_text())["epochs"]
    keys = ("window_days", "epoch", "optimizer_steps", "train_loss_mean",
            "train_loss_per_batch", "validation_loss", "checkpoint")
    assert [{key: entry[key] for key in keys} for entry in log["epochs"]] \
        == [{key: entry[key] for key in keys} for entry in reference]
    for entry in reference:
        name = entry["checkpoint"]["path"]
        assert driver._sha256(tmp_path / "run" / name) \
            == driver._sha256(one_window["folder"] / "run" / name)


# ------------------------------------------------ widened template target filter

def test_every_committed_training_config_states_its_target_filter_width():
    widths = {path.name: driver.load_config(path)["dataset"][
        "target_filter_sigma_cells"]
        for path in sorted((_ROOT / "config/swg_online").glob("*.json"))
        if "network" in json.loads(path.read_text())
        and path.name not in PENDING_CONFIGS}
    assert widths.pop(EXP6[0].name) == 1.5
    assert widths.pop(EXP6[1].name) == 2.0
    assert len(widths) >= 17 and set(widths.values()) == {1.0}


@pytest.mark.parametrize("path,experiment,sigma,shown", [
    (EXP6[0], "swgonl_exp6_rigidlid_template_sigma1p5", 1.5, "1.5"),
    (EXP6[1], "swgonl_exp6_rigidlid_template_sigma2", 2.0, "2")])
def test_exp6_configs_differ_from_exp5_template_only_in_the_filter_width(
        path, experiment, sigma, shown):
    base, new = driver.load_config(EXP5_TEMPLATE), driver.load_config(path)
    prefix = (f"Wider target filter (user 2026-10-18): template objective "
              f"with the Gaussian target smoothed to sigma = {shown} coarse "
              "cells by a second coarse-grid pass; otherwise exp5 template. ")
    stale = " The dataset roots below are PLACEHOLDERS"
    assert "PLACEHOLDERS" in base["description"]
    assert new["description"].startswith(prefix + "Speed-up of exp4")
    assert "PLACEHOLDERS" not in new["description"]
    assert new["description"] == prefix + base["description"].split(stale)[0]
    assert new == base | {
        "experiment": experiment, "description": new["description"],
        "dataset": base["dataset"] | {"target_filter_sigma_cells": sigma}}
    assert new["objective"] == "template" and new["surface"] == "rigid_lid"


def test_contract_names_the_target_filter_width_and_check_data_reads_it(
        tmp_path, small_card, record):
    for sigma in (1.0, 1.5):
        config = _config(record)
        config["dataset"]["target_filter_sigma_cells"] = sigma
        payload = driver.write_dataset_contract(
            config, tmp_path / f"contract_{sigma}.json")
        assert payload["target_filter_sigma_cells"] == sigma
        contract = driver.SWGDatasetContract.from_json(
            tmp_path / f"contract_{sigma}.json", card=small_card)
        assert contract.target_filter_sigma_cells == sigma
        assert (contract._target_filter is None) == (sigma == 1.0)
    # a wider filter is a different contract, hence a different run identity
    assert driver._sha256(tmp_path / "contract_1.0.json") \
        != driver._sha256(tmp_path / "contract_1.5.json")
    config["dataset"]["target_filter_sigma_cells"] = 1.5
    report = driver.check_data(config, card=small_card)
    assert len(report["batches"]) == 6


def test_committed_long_config_extends_exp4_symmetric_and_refuses_what_does_not_fit(
        tmp_path):
    """The long rigid-lid configuration is exp4 symmetric with the 16- and
    32-day windows appended; a curriculum past eight days must start with
    1/2/4/8 and fit the splits."""
    base = driver.load_config(EXP4_SYMMETRIC)
    long = driver.load_config(EXP4_SYMMETRIC_LONG)
    assert base["curriculum"]["window_days"] == [1, 2, 4, 8]
    assert long["curriculum"]["window_days"] == [1, 2, 4, 8, 16, 32]
    assert long | {"description": base["description"],
                   "curriculum": base["curriculum"]} == base
    for name, edit, match in (
            ("no prefix", lambda c: c["curriculum"].update(
                window_days=[8, 16]), "without starting with"),
            ("unordered", lambda c: c["curriculum"].update(
                window_days=[1, 2, 4, 8, 32, 16]), "window_days must increase"),
            ("no room for a batch", lambda c: c["dataset"]["splits_days"].update(
                validation=[4265, 4265 + 9 * 32]),
             "validation split holds 9 32-day")):
        config = copy.deepcopy(long)
        edit(config)
        path = tmp_path / f"{name.replace(' ', '_')}.json"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match=match):
            driver.load_config(path)


# ------------------------------- arms B and C: noise channel, energy score, statistics

EXP7 = _ROOT / "config/swg_online/exp7_rigidlid_energy_score_m8.json"
EXP8 = _ROOT / "config/swg_online/exp8_rigidlid_window_stats.json"
EXP9 = _ROOT / "config/swg_online/exp9_rigidlid_neuralgcm.json"
EXP10 = _ROOT / "config/swg_online/exp10_rigidlid_energy_score_neuralgcm_m8.json"
# Shipped with a placeholder cutoff table (the twin measurement is pending):
# they refuse to load, so the "every committed config" tests leave them out.
PENDING_CONFIGS = (EXP9.name, EXP10.name)
W08_CHECKPOINT = (
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/"
    "round17_rigidlid/swgonl_exp4_rigidlid_symmetric_ring_weighted_fp32/"
    "seed20261013/checkpoints/w08_e05.eqx")


def test_every_committed_training_config_states_its_loss_and_noise_channel():
    trained_configs = [
        path for path in sorted((_ROOT / "config/swg_online").glob("*.json"))
        if "network" in json.loads(path.read_text())
        and path.name not in PENDING_CONFIGS]
    seen = {path.name: (driver.load_config(path)["loss"],
                        driver.load_config(path)["network"]["noise_channel"],
                        driver.load_config(path)["network"][
                            "initial_acceleration_gate"])
            for path in trained_configs}
    # The arms load the trained w08 file, whose closure is 10-15 x the wind
    # acceleration; every other configuration keeps the 1% bar enforced.
    assert seen.pop(EXP7.name) == (
        {"kind": "energy_score", "members": 8}, "white_input", "recorded_only")
    assert seen.pop(EXP8.name) == (
        {"kind": "eq7_window_statistics", "calibration_window_days": 8}, "none",
        "recorded_only")
    assert len(seen) >= 17
    assert {json.dumps(value) for value in seen.values()} == {
        json.dumps(({"kind": "eq7"}, "none", "enforced"))}


@pytest.mark.parametrize("path,loss", [
    (EXP7, {"kind": "energy_score", "members": 8}),
    (EXP8, {"kind": "eq7_window_statistics", "calibration_window_days": 8})])
def test_committed_arm_configs_differ_from_the_long_config_only_in_the_arm(
        path, loss):
    long = driver.load_config(EXP4_SYMMETRIC_LONG)
    arm = driver.load_config(path)
    differing = sorted(key for key in long if long[key] != arm[key])
    assert differing == sorted([
        "experiment", "description", "network", "loss", "seeds", "dataset",
        "curriculum"])
    assert arm["curriculum"] == long["curriculum"] | {"window_days": [8, 16, 32]}
    assert arm["loss"] == loss
    assert arm["network"]["initial_checkpoint"] == W08_CHECKPOINT
    arm_keys = ("noise_channel", "initial_checkpoint",
                "initial_acceleration_gate")
    assert {key: value for key, value in arm["network"].items()
            if key not in arm_keys} \
        == {key: value for key, value in long["network"].items()
            if key not in arm_keys}
    assert arm["network"]["initial_acceleration_gate"] == "recorded_only"
    assert arm["seeds"] == [20261013]
    assert arm["dataset"]["cache"] == "memory"
    assert arm["dataset"] | {"cache": "none"} == long["dataset"]
    assert arm["gradient_checkpointing"] is True
    # Warm start (user decision 2026-10-19): from the w08 checkpoint's window.
    assert arm["curriculum"]["window_days"] == [8, 16, 32]
    assert (arm["surface"], arm["objective"], arm["training_precision"],
            arm["target_interval_hours"]) == (
        "rigid_lid", "symmetric", "fp32", 3)


def _fake_run(tmp_path, windows=(1, 2, 4, 8), log=True):
    """A run directory whose training log lists w<NN>_e01 for each window."""
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    if log:
        (run / "training_log.json").write_text(json.dumps({"epochs": [
            {"window_days": w, "checkpoint": {
                "path": f"checkpoints/w{w:02d}_e01.eqx"}} for w in windows]}))
    return run / "checkpoints" / "w08_e01.eqx"


@pytest.mark.parametrize("windows,match", [
    ([8, 16, 32], None),
    ([8, 16], None),
    ([16, 32], "must start at the initial checkpoint's last trained window, "
               "8 days"),
    ([1, 2, 4, 8, 16, 32], None),     # the cold-start rule still holds
    ([4, 8, 16], "must start at the initial checkpoint's last trained"),
    ([8, 6], "window_days must increase")])
def test_warm_start_curriculum_starts_at_the_checkpoints_last_window(
        tmp_path, windows, match):
    config = copy.deepcopy(driver.load_config(EXP8))
    config["network"]["initial_checkpoint"] = str(_fake_run(tmp_path))
    config["curriculum"]["window_days"] = windows
    path = tmp_path / "warm.json"
    path.write_text(json.dumps(config))
    if match is None:
        assert driver.load_config(path)["curriculum"]["window_days"] == windows
    else:
        with pytest.raises(ValueError, match=match):
            driver.load_config(path)


def test_cold_start_curriculum_beyond_eight_days_still_needs_the_prefix(tmp_path):
    config = copy.deepcopy(driver.load_config(EXP4_SYMMETRIC_LONG))
    assert "initial_checkpoint" not in config["network"]
    config["curriculum"]["window_days"] = [8, 16, 32]
    path = tmp_path / "cold.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="without starting with"):
        driver.load_config(path)


@pytest.mark.parametrize("kwargs,match", [
    ({"log": False}, "run log .* cannot be read"),
    ({"windows": (1, 2, 4)}, "lists 0 epochs with checkpoint checkpoints/w08"),
])
def test_warm_start_refuses_an_unreadable_checkpoint_run(
        tmp_path, kwargs, match):
    config = copy.deepcopy(driver.load_config(EXP8))
    config["network"]["initial_checkpoint"] = str(_fake_run(tmp_path, **kwargs))
    config["curriculum"]["window_days"] = [8, 16, 32]
    path = tmp_path / "unreadable.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=match):
        driver.load_config(path)


@pytest.mark.parametrize("edit,match", [
    (lambda c: c["network"].update(noise_channel="gaussian"),
     "noise_channel must be one of"),
    (lambda c: c["network"].pop("noise_channel"), "must hold exactly the keys"),
    (lambda c: c.pop("loss"), "must hold exactly the keys"),
    (lambda c: c.update(loss={"kind": "crps"}), "loss kind must be one of"),
    (lambda c: c.update(loss={"kind": "eq7", "members": 8}),
     "loss must hold exactly the keys"),
    (lambda c: c.update(loss={"kind": "energy_score", "members": 1}),
     "members must be an integer >= 2"),
    (lambda c: c.update(loss={"kind": "energy_score", "members": 8}),
     "energy_score loss needs network noise_channel"),
    (lambda c: c.update(loss={"kind": "eq7_window_statistics",
                              "calibration_window_days": 3}),
     "calibration_window_days must be one of the curriculum"),
    (lambda c: c.update(loss={"kind": "eq7_window_statistics"}),
     "loss must hold exactly the keys"),
    (lambda c: c["network"].update(noise_channel="white_input"),
     "energy_score loss needs network noise_channel"),
    (lambda c: c["network"].update(initial_acceleration_gate="off"),
     "initial_acceleration_gate must be one of"),
    (lambda c: c["network"].pop("initial_acceleration_gate"),
     "must hold exactly the keys"),
    (lambda c: c["network"].update(initial_acceleration_gate="recorded_only"),
     "needs network initial_checkpoint")])
def test_load_config_refuses_an_inconsistent_loss_or_noise_channel(
        tmp_path, edit, match):
    config = copy.deepcopy(driver.load_config(EXP4_SYMMETRIC))
    edit(config)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=match):
        driver.load_config(path)


@pytest.fixture(scope="module")
def lid_geometry(lid_card):
    return build_closure_geometry(lid_card)


def _lid_batch(lid_card, windows=2, hours=8):
    rng = np.random.default_rng(4)
    template = lid_card.recipe.initial_state
    u_shape = tuple(template.u.data[:, 1:, 0].shape)
    v_shape = tuple(template.v.data[1:, :, 0].shape)

    def native(*lead):
        return (np.zeros(lead + tuple(template.eta.data.shape)),
                1.0e-3 * rng.normal(size=lead + u_shape),
                1.0e-3 * rng.normal(size=lead + v_shape))

    return (tuple(map(jnp.asarray, native(windows))),
            tuple(map(jnp.asarray, native(windows, hours))))


def _programs(lid_card, lid_geometry, loss, constants, **changes):
    config = driver.load_config(EXP4_SYMMETRIC)
    return driver.build_batch_programs(
        lid_card, lid_geometry, window_days=1,
        target_interval_hours=config["target_interval_hours"],
        objective="symmetric", wall_ring=config["wall_ring"],
        training_precision="fp64", checkpoint=True, loss=loss,
        constants=constants, **changes)


def _channel(lid_card, lid_geometry, scale, seed=3):
    config = driver.load_config(EXP7)
    spec, _ = driver.network_spec(config, lid_card, lid_geometry)
    four = SWGScalarPotentialCNN(
        spec._replace(noise_channel=False), key=jax.random.PRNGKey(seed))
    return swg_online.add_noise_channel(
        four, key=jax.random.PRNGKey(seed + 1), relative_scale=scale)


def test_energy_score_programs_run_members_that_differ_and_every_leaf_is_live(
        lid_card, lid_geometry):
    network = _channel(lid_card, lid_geometry, 1.0)
    loss = {"kind": "energy_score", "members": 3}
    normalizers = swg_online.BalancedNormalizers(1.0e-10, 1.0e-6, 1.0e-6)
    programs = _programs(lid_card, lid_geometry, loss, normalizers)
    initial, target = _lid_batch(lid_card)
    key = jax.random.PRNGKey(1)
    value, grads = programs["value_and_grad"](network, initial, target, key)
    assert np.isfinite(float(value)) and float(value) > 0.0
    assert_network_gradients_live(grads)        # includes the xi weights
    assert np.any(np.asarray(grads.layers[0].weight[:, 4:]) != 0.0)
    score, member_truth, spread = (float(x) for x in programs["report"](
        network, initial, target, key))
    assert score == pytest.approx(float(value), rel=1.0e-12)
    assert score == pytest.approx(member_truth - 0.5 * spread, rel=1.0e-9)
    assert spread > 1.0e-6 * member_truth       # the members differ
    # Seeded: the same key gives the same ensemble; another key another one.
    again = float(programs["loss"](network, initial, target, key))
    other = float(programs["loss"](network, initial, target,
                                   jax.random.PRNGKey(2)))
    assert again == float(value) and other != again
    # Without xi weights the members coincide: no spread (the monitor reads 0).
    silent = _channel(lid_card, lid_geometry, 0.0)
    assert float(programs["report"](silent, initial, target, key)[2]) \
        < 1.0e-6 * spread
    # Only the energy score takes members; the Eq. 7 loss takes no constants.
    with pytest.raises(ValueError, match="takes no loss constants"):
        _programs(lid_card, lid_geometry, {"kind": "eq7"}, normalizers)
    with pytest.raises(ValueError, match="loss kind"):
        _programs(lid_card, lid_geometry, {"kind": "crps"}, normalizers)


def test_window_statistics_programs_with_zero_weights_are_eq7_bit_for_bit(
        precision_steps):
    """Against the parent code path (``_parent_batch_functions``, written
    before the loss choice existed): loss and every gradient leaf equal."""
    s = precision_steps
    config = driver.load_config(EXP1_SYMMETRIC)
    loss = {"kind": "eq7_window_statistics", "calibration_window_days": 1}
    options = dict(
        window_days=1, target_interval_hours=1, objective=config["objective"],
        wall_ring=config["wall_ring"], training_precision="fp64",
        checkpoint=True)
    zero = driver.build_batch_programs(
        s["card"], s["geometry"], loss=loss, constants=(0.0, 0.0), **options)
    eq7 = driver.build_batch_programs(
        s["card"], s["geometry"], loss={"kind": "eq7"}, constants=None, **options)
    for programs in (zero, eq7):
        value, grads = driver.batch_value_and_grad(
            programs["value_and_grad"], s["network"], s["batch"], s["key"], 10)
        assert float(value) == float(s["parent"]["loss"]) > 0.0
        assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(
            jax.tree.leaves(grads), jax.tree.leaves(s["parent"]["grads"]),
            strict=True))
    # Weighted: the total is Eq. 7 plus the weighted terms the report shows.
    total, eq7_term, vorticity, kinetic = driver.batch_report(
        zero["report"], s["network"], s["batch"], s["key"], 10)
    assert total == eq7_term == float(s["parent"]["loss"])
    assert vorticity > 0.0 and kinetic > 0.0
    weighted = driver.build_batch_programs(
        s["card"], s["geometry"], loss=loss, constants=(2.0, 3.0), **options)
    value, grads = driver.batch_value_and_grad(
        weighted["value_and_grad"], s["network"], s["batch"], s["key"], 10)
    assert float(value) == pytest.approx(
        eq7_term + 2.0 * vorticity + 3.0 * kinetic, rel=1.0e-12)
    assert_network_gradients_live(grads)
    assert float(value) != float(s["parent"]["loss"])


def test_free_run_with_the_noise_channel_draws_fresh_seeded_xi_each_step(
        lid_card, lid_geometry):
    key = jax.random.PRNGKey(7)
    network = _channel(lid_card, lid_geometry, 1.0)
    forcing, one_day = driver.free_run_day(
        network, lid_card, lid_geometry, key=key)
    start = driver._double_gyre_driver().seed_steady_forcing_carry(
        lid_card.recipe.initial_state, forcing)
    stage_noise = jnp.zeros((3,) + tuple(lid_geometry.tmask.shape))
    per_day = driver.steps_per_day(lid_card)
    assert per_day == 8

    def steps(first, step_key):
        state = start
        for index in range(first, first + per_day):
            state = driver.swg_online_step(
                network, state, stage_noise, step_key(index), jnp.asarray(0.0),
                card=lid_card, geometry=lid_geometry, forcing=forcing)
        return np.asarray(state.u.data)

    # The compiled day is eight steps, step n keyed fold_in(key, n) ...
    day, _ = one_day(start)
    np.testing.assert_allclose(
        np.asarray(day.u.data), steps(0, lambda n: jax.random.fold_in(key, n)),
        rtol=1.0e-10, atol=1.0e-14)
    # ... a later day continues the stream (steps 8..15) rather than repeating it
    later, _ = one_day(start, 1)
    np.testing.assert_allclose(
        np.asarray(later.u.data),
        steps(per_day, lambda n: jax.random.fold_in(key, n)),
        rtol=1.0e-10, atol=1.0e-14)
    scale = np.abs(np.asarray(day.u.data)).max()
    assert np.abs(np.asarray(later.u.data) - np.asarray(day.u.data)).max() \
        > 1.0e-3 * scale
    # ... and a frozen step key (the same xi every step) is another day.
    assert np.abs(steps(0, lambda n: key) - np.asarray(day.u.data)).max() \
        > 1.0e-3 * scale
    # Seeded: the same key gives the same day; another key another.
    repeat, _ = one_day(start)
    assert np.array_equal(np.asarray(day.u.data), np.asarray(repeat.u.data))
    _, other_day = driver.free_run_day(
        network, lid_card, lid_geometry, key=jax.random.PRNGKey(8))
    assert np.abs(np.asarray(other_day(start)[0].u.data)
                  - np.asarray(day.u.data)).max() > 1.0e-3 * scale


class _Field(NamedTuple):
    data: jax.Array


class _State(NamedTuple):
    eta: _Field
    u: _Field
    v: _Field
    rigid_lid_phi: object = None


class _IndexModel:
    """Adds 0.5 to eta per step, or the step index when it is given."""

    def step(self, state, dt_s, surface_forcing, step_index=None):
        added = 0.5 if step_index is None else step_index.astype(jnp.float64)
        return state._replace(eta=_Field(state.eta.data + added))


def test_day_mean_stepper_passes_global_step_indices_only_when_asked():
    state = _State(_Field(jnp.zeros((4, 4))), _Field(jnp.zeros((4, 5, 1))),
                   _Field(jnp.zeros((5, 4, 1))))
    stepper = driver._double_gyre_driver().build_scanned_day_mean_stepper(
        _IndexModel(), 1.0, None, 3)
    plain, _ = stepper(state)
    assert np.all(np.asarray(plain.eta.data) == 1.5)        # no index passed
    indexed, means = stepper(state, jnp.asarray(10, dtype=jnp.int32))
    assert np.all(np.asarray(indexed.eta.data) == 10 + 11 + 12)
    # The daily mean sees each post-step eta: 10, 21, 33.
    assert np.allclose(np.asarray(means[0]), (10 + 21 + 33) / 3.0)


def _textured(card):
    """A day edit that gives the synthetic record's velocities a smooth
    spatial structure.  The plain record's flow is uniform inside the basin
    (balanced vorticity and rotational velocity identically zero, so no
    variance and no statistic to measure); the same factor on every chunk and
    on the start arrays keeps overlapping hours equal."""
    ny, nx = np.asarray(card.recipe.land_mask).shape
    factor = 1.0 + 0.5 * np.sin(2.0 * np.pi * np.arange(nx) / nx)[None, :] \
        * np.cos(2.0 * np.pi * np.arange(ny) / ny)[:, None]

    def edit(day, arrays):
        for name in list(arrays):
            if "_u_" in name or "_v_" in name:
                arrays[name] = arrays[name] * factor
    return edit


@pytest.fixture(scope="module")
def textured_record(tmp_path_factory, small_card):
    root = tmp_path_factory.mktemp("swg_textured") / "record"
    geometry = build_closure_geometry(small_card)
    masks = (np.asarray(geometry.umask), np.asarray(geometry.vmask))
    for first, days in ((0, 11), (11, 11), (52, 20), (102, 20)):
        records._write_chunk(
            root / f"dual_day{first:04d}_to_day{first + days:04d}",
            source_day=first, days=days, masks=masks,
            truth_run=root.parent / "truth", day_edit=_textured(small_card))
    return root


def _arm_config(record, loss, noise_channel, **changes):
    config = _config(record, **changes)
    config["curriculum"].update(window_days=[1], epochs_per_window=2)
    config["network"]["noise_channel"] = noise_channel
    config["loss"] = loss
    return config


ENERGY_SEED = 11


@pytest.fixture(scope="module")
def energy_run(tmp_path_factory, small_card, textured_record, trained):
    folder = tmp_path_factory.mktemp("swg_energy")
    config = _arm_config(textured_record, {"kind": "energy_score", "members": 2},
                         "white_input")
    # As in production: the file of a trained four-channel network, whose
    # closure is far above the random-initialisation bar (recorded only).
    config["network"]["initial_checkpoint"] = str(
        trained["folder"] / "run" / "checkpoints" / "w01_e02.eqx")
    config["network"]["initial_acceleration_gate"] = "recorded_only"
    path = _write_config(folder / "config.json", config)
    log = driver.train(config, path, card=small_card, seed=ENERGY_SEED,
                       run_dir=folder / "run", device=CPU,
                       windows_per_call=10, commit="0" * 40)
    return {"folder": folder, "config": config, "path": path, "log": log}


def test_energy_score_run_freezes_normalizers_and_logs_the_collapse_monitor(
        energy_run, small_card):
    run = energy_run["folder"] / "run"
    log = energy_run["log"]
    assert log["status"] == "complete"
    assert log["gates"]["first_step_no_inert"]["passed"]
    manifest = json.loads((run / "run_manifest.json").read_text())
    constants = manifest["loss_constants"]
    assert constants["kind"] == "energy_score"
    assert all(value > 0.0 for value in constants["normalizers"].values())
    assert constants["source"]["split"] == "train"
    assert constants["source"]["window_days"] == 1
    assert manifest["network"]["input_channels"][-1].startswith("xi ~ N(0,1)")
    assert manifest["network"]["spec"]["noise_channel"] is True
    assert manifest["loss"]["members"] == 2
    for entry in log["epochs"]:
        report = entry["validation_report"]
        assert report["energy_score"] == entry["validation_loss"]
        assert report["member_spread"] > 0.0
        assert report["spread_over_member_truth_distance"] == pytest.approx(
            report["member_spread"] / report["member_truth_distance"])
        assert np.isfinite(entry["train_loss_mean"])
    # The normalizers are the training split's own: recompute independently.
    contract = driver.SWGDatasetContract.from_json(
        run / "dataset_contract.json", card=small_card)
    geometry = build_closure_geometry(small_card)
    moments = np.zeros((3, 3))
    for batch in contract.batches(
            split="train", window_days=1, epoch=0, seed=ENERGY_SEED,
            target_interval_hours=1):
        moments += swg_online.balanced_field_second_moments(
            swg_online.diagnose_native_windows(
                tuple(jnp.asarray(v) for v in batch.target), geometry,
                objective=energy_run["config"]["objective"]), geometry)
    expected = swg_online.normalizers_from_moments(moments)
    assert tuple(constants["normalizers"].values()) == pytest.approx(
        tuple(expected), rel=1.0e-9)


def test_energy_score_run_resumes_with_the_recorded_normalizers(
        energy_run, small_card, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(energy_run["folder"] / "run", run)
    before = json.loads((run / "run_manifest.json").read_text())
    log = driver.train(
        energy_run["config"], energy_run["path"], card=small_card,
        seed=ENERGY_SEED, run_dir=run, device=CPU, windows_per_call=10,
        commit="0" * 40)
    assert log["status"] == "complete"
    assert json.loads((run / "run_manifest.json").read_text()) == before
    # Epochs but no recorded constants: refused, never silently re-measured.
    stripped = {key: value for key, value in before.items()
                if key != "loss_constants"}
    (run / "run_manifest.json").write_text(json.dumps(stripped))
    with pytest.raises(RuntimeError, match="records no loss constants"):
        driver.frozen_loss_constants(
            energy_run["config"], run / "run_manifest.json", have_epochs=True)
    # A manifest of another loss is refused too.
    wrong = before | {"loss_constants": {"kind": "eq7_window_statistics"}}
    (run / "run_manifest.json").write_text(json.dumps(wrong))
    with pytest.raises(RuntimeError, match="loss constants are for"):
        driver.frozen_loss_constants(
            energy_run["config"], run / "run_manifest.json", have_epochs=True)


def test_energy_score_free_run_is_admitted_with_the_channel_in_its_spec(
        energy_run, small_card):
    run_dir = energy_run["folder"] / "run"
    manifest = driver.run_free_run(
        energy_run["config"], energy_run["path"], card=small_card,
        seed=ENERGY_SEED, run_dir=run_dir, window_days=1, device=CPU,
        commit="0" * 40)
    closure = manifest["closure"]
    assert closure["network_spec"]["noise_channel"] is True
    assert "white-noise input channel" in manifest["integration"]
    assert manifest["numerical_health"]["all_finite"]
    assert swg_online.admit_free_run_closure(manifest) == closure
    # Manifests written before the channel existed lack the key: still admitted.
    legacy = copy.deepcopy(manifest)
    del legacy["closure"]["network_spec"]["noise_channel"]
    assert swg_online.admit_free_run_closure(legacy)
    extra = copy.deepcopy(manifest)
    extra["closure"]["network_spec"]["amplitude"] = 1.0
    with pytest.raises(ValueError, match="network_spec must hold exactly"):
        swg_online.admit_free_run_closure(extra)


@pytest.fixture(scope="module")
def statistics_run(tmp_path_factory, small_card, textured_record):
    folder = tmp_path_factory.mktemp("swg_statistics")
    config = _arm_config(
        textured_record,
        {"kind": "eq7_window_statistics", "calibration_window_days": 1},
        "none")
    path = _write_config(folder / "config.json", config)
    log = driver.train(config, path, card=small_card, seed=11,
                       run_dir=folder / "run", device=CPU,
                       windows_per_call=10, commit="0" * 40)
    return {"folder": folder, "config": config, "path": path, "log": log}


def test_window_statistics_run_sets_the_weights_once_from_the_first_batch(
        statistics_run, small_card):
    run = statistics_run["folder"] / "run"
    log = statistics_run["log"]
    assert log["status"] == "complete"
    manifest = json.loads((run / "run_manifest.json").read_text())
    constants = manifest["loss_constants"]
    calibration = constants["calibration"]
    # Each added term equals the Eq. 7 term on the first batch.
    assert constants["vorticity_weight"] * calibration["vorticity_nrmse2"] \
        == pytest.approx(calibration["eq7"], rel=1.0e-12)
    assert constants["eke_weight"] * calibration["eke_nrmse2"] \
        == pytest.approx(calibration["eq7"], rel=1.0e-12)
    # ... recomputed independently on the contract's first batch and the
    # seed's network.
    config = statistics_run["config"]
    contract = driver.SWGDatasetContract.from_json(
        run / "dataset_contract.json", card=small_card)
    batch = next(contract.batches(split="train", window_days=1, epoch=0,
                                  seed=11, target_interval_hours=1))
    assert calibration["start_absolute_hours"] == [
        int(hour) for hour in batch.start_absolute_hour]
    geometry = build_closure_geometry(small_card)
    spec, _ = driver.network_spec(config, small_card, geometry)
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(11))
    programs = driver.build_batch_programs(
        small_card, geometry, window_days=1, target_interval_hours=1,
        objective=config["objective"], wall_ring=config["wall_ring"],
        training_precision="fp64", checkpoint=True, loss=config["loss"],
        constants=(0.0, 0.0))
    _, eq7, vorticity, kinetic = driver.batch_report(
        programs["report"], network, batch, jax.random.PRNGKey(11), 10)
    assert calibration["eq7"] == pytest.approx(eq7, rel=1.0e-9)
    assert calibration["vorticity_nrmse2"] == pytest.approx(vorticity, rel=1.0e-9)
    assert calibration["eke_nrmse2"] == pytest.approx(kinetic, rel=1.0e-9)
    # Frozen: every logged epoch's total is built from those same weights.
    for entry in log["epochs"]:
        report = entry["validation_report"]
        assert report["total"] == pytest.approx(
            report["eq7"] + constants["vorticity_weight"]
            * report["vorticity_nrmse2"] + constants["eke_weight"]
            * report["eke_nrmse2"], rel=1.0e-9)
        assert report["total"] == pytest.approx(entry["validation_loss"], rel=1.0e-12)
    assert manifest["loss"]["weights"].startswith("set once")
    assert log["gates"]["first_step_no_inert"]["passed"]


def test_window_statistics_run_resumes_with_the_recorded_weights(
        statistics_run, small_card, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(statistics_run["folder"] / "run", run)
    before = json.loads((run / "run_manifest.json").read_text())
    driver.train(
        statistics_run["config"], statistics_run["path"], card=small_card,
        seed=11, run_dir=run, device=CPU, windows_per_call=10, commit="0" * 40)
    assert json.loads((run / "run_manifest.json").read_text()) == before
    # The recorded keys are not part of the run's identity.
    assert driver.run_identity(before) == {
        key: value for key, value in before.items() if key != "loss_constants"}


def test_noise_channel_run_starts_from_a_four_channel_checkpoint(
        tmp_path, small_card, record, trained):
    """A finished deterministic network's file seeds the channel network:
    trained leaves identical, xi weights at 1e-2 of the SSH-channel rms."""
    checkpoint = trained["folder"] / "run" / "checkpoints" / "w01_e02.eqx"
    config = _arm_config(
        record, {"kind": "energy_score", "members": 2}, "white_input")
    config["network"]["initial_checkpoint"] = str(checkpoint)
    geometry = build_closure_geometry(small_card)
    spec, _ = driver.network_spec(config, small_card, geometry)
    optimizer = create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))
    key = jax.random.PRNGKey(11)
    network, opt_state = driver.initial_network(config, spec, optimizer, key=key)
    four_spec = spec._replace(noise_channel=False)
    four = SWGScalarPotentialCNN(four_spec, key=key)
    loaded, _ = driver.load_checkpoint_or_fail(
        checkpoint, (four, optimizer.init(eqx.filter(four, eqx.is_inexact_array))),
        what="test")
    assert network.layers[0].weight.shape[1] == 5
    assert np.array_equal(np.asarray(network.layers[0].weight[:, :4]),
                          np.asarray(loaded.layers[0].weight))
    rms = float(jnp.sqrt(jnp.mean(loaded.layers[0].weight[:, 0] ** 2)))
    xi_weights = np.asarray(network.layers[0].weight[:, 4:])
    assert 0.3e-2 * rms < xi_weights.std() < 3.0e-2 * rms
    assert eqx.tree_equal(opt_state, optimizer.init(
        eqx.filter(network, eqx.is_inexact_array)))
    # The identity records how the channel was grown.
    assert driver.NOISE_CHANNEL_INIT_RELATIVE_SCALE == 1.0e-2
    # A wrong file is refused before anything is built.
    config["network"]["initial_checkpoint"] = str(tmp_path / "missing.eqx")
    with pytest.raises(FileNotFoundError):
        driver.initial_network(config, spec, optimizer, key=key)


def test_a_loaded_network_may_start_past_the_acceleration_bar_only_when_recorded(
        tmp_path, small_card, record):
    """The bar is for a seed's random initialisation: a network loaded from a
    file whose closure is far above it is refused when the gate is
    enforced and starts, with the measurement logged, when recorded_only."""
    config = _arm_config(record, {"kind": "eq7"}, "none")
    geometry = build_closure_geometry(small_card)
    spec, _ = driver.network_spec(config, small_card, geometry)
    optimizer = create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(11))
    loud = eqx.tree_at(lambda n: n.layers[-1].weight, network,
                       network.layers[-1].weight * 100.0)
    checkpoint = tmp_path / "loud.eqx"
    driver.save_checkpoint(
        (loud, optimizer.init(eqx.filter(loud, eqx.is_inexact_array))), checkpoint)
    config["network"]["initial_checkpoint"] = str(checkpoint)
    config["curriculum"].update(epochs_per_window=1)
    path = _write_config(tmp_path / "enforced.json", config)
    with pytest.raises(RuntimeError, match="initial closure acceleration"):
        driver.train(config, path, card=small_card, seed=11,
                     run_dir=tmp_path / "enforced", device=CPU,
                     windows_per_call=10, commit="0" * 40)
    config["network"]["initial_acceleration_gate"] = "recorded_only"
    path = _write_config(tmp_path / "recorded.json", config)
    log = driver.train(config, path, card=small_card, seed=11,
                       run_dir=tmp_path / "recorded", device=CPU,
                       windows_per_call=10, commit="0" * 40)
    gate = log["gates"]["initial_closure_acceleration"]
    assert log["status"] == "complete" and len(log["epochs"]) == 1
    assert gate["enforced"] is False and gate["passes"] is False
    assert gate["maximum_ratio"] > 0.1


# --------------------- NeuralGCM-style losses: exp9 (Eq. 7) and exp10 (energy)

FULL_BAND_TABLE = {"lead_hours": [0], "cutoff_band": [1000]}
# 3-hour records of a 1-day window: leads 3 | 6, 9 | 12 .. 24.
LID_TABLE = {"lead_hours": [0, 6, 12], "cutoff_band": [1000, 8, 4]}
PENDING = "PENDING_TWIN_MEASUREMENT"


def _with_table(path, tmp_path, table=LID_TABLE):
    config = json.loads(path.read_text())
    assert config["loss"]["lead_time_cutoff"] == PENDING
    config["loss"]["lead_time_cutoff"] = table
    target = tmp_path / path.name
    target.write_text(json.dumps(config))
    return target


@pytest.mark.parametrize("path", [EXP9, EXP10])
def test_committed_neuralgcm_configs_refuse_to_load_until_the_table_is_measured(
        path, tmp_path):
    assert json.loads(path.read_text())["loss"]["lead_time_cutoff"] == PENDING
    with pytest.raises(ValueError, match="PENDING_TWIN_MEASUREMENT"):
        driver.load_config(path)
    # With a measured table in its place the configuration loads.
    assert driver.load_config(_with_table(path, tmp_path))[
        "loss"]["lead_time_cutoff"] == LID_TABLE


def test_exp9_is_the_exp4_symmetric_recipe_from_scratch_with_the_new_loss(
        tmp_path):
    parent = driver.load_config(EXP4_SYMMETRIC)
    arm = driver.load_config(_with_table(EXP9, tmp_path))
    assert sorted(key for key in parent if parent[key] != arm[key]) == [
        "dataset", "description", "experiment", "loss", "seeds"]
    assert arm["loss"] == {"kind": "eq7_neuralgcm", "calibration_window_days": 1,
                           "lead_time_cutoff": LID_TABLE}
    assert arm["seeds"] == parent["seeds"][:1] == [20261013]
    assert arm["dataset"] == parent["dataset"] | {"cache": "memory"}
    # From scratch: no checkpoint, the 1% bar enforced, the 1, 2, 4, 8 days.
    assert "initial_checkpoint" not in arm["network"]
    assert arm["network"] == parent["network"]
    assert arm["curriculum"] == parent["curriculum"]
    assert arm["curriculum"]["window_days"] == [1, 2, 4, 8]


def test_exp10_is_exp7_with_only_the_loss_changed(tmp_path):
    parent = driver.load_config(EXP7)
    arm = driver.load_config(_with_table(EXP10, tmp_path))
    assert sorted(key for key in parent if parent[key] != arm[key]) == [
        "description", "experiment", "loss"]
    assert arm["loss"] == {
        "kind": "energy_score_neuralgcm", "members": 8,
        "calibration_window_days": 8, "lead_time_cutoff": LID_TABLE}
    assert arm["loss"]["members"] == parent["loss"]["members"]


@pytest.mark.parametrize("edit,match", [
    (lambda c: c["loss"].pop("lead_time_cutoff"),
     "loss must hold exactly the keys"),
    (lambda c: c["loss"].update(lead_time_cutoff=PENDING),
     "PENDING_TWIN_MEASUREMENT"),
    (lambda c: c["loss"].update(lead_time_cutoff={"lead_hours": [0]}),
     "lead_time_cutoff must be"),
    (lambda c: c["loss"].update(
        lead_time_cutoff={"lead_hours": [0, 6], "cutoff_band": [4, 8]}),
     "non-increasing"),
    (lambda c: c["loss"].update(calibration_window_days=3),
     "calibration_window_days must be one of the curriculum"),
    (lambda c: c["loss"].update(members=8), "loss must hold exactly the keys"),
    (lambda c: c["network"].update(noise_channel="white_input"),
     "energy_score loss needs network noise_channel"),
    (lambda c: c["loss"].update(kind="eq7_neuralgcm_v2"),
     "loss kind must be one of")])
def test_load_config_refuses_an_inconsistent_eq7_neuralgcm_loss(
        tmp_path, edit, match):
    config = copy.deepcopy(driver.load_config(_with_table(EXP9, tmp_path)))
    edit(config)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=match):
        driver.load_config(path)


@pytest.mark.parametrize("edit,match", [
    (lambda c: c["loss"].pop("lead_time_cutoff"),
     "loss must hold exactly the keys"),
    (lambda c: c["loss"].pop("members"), "loss must hold exactly the keys"),
    (lambda c: c["loss"].update(members=1), "members must be an integer >= 2"),
    (lambda c: c["network"].update(noise_channel="none"),
     "energy_score loss needs network noise_channel"),
    (lambda c: c["loss"].update(lead_time_cutoff=PENDING),
     "PENDING_TWIN_MEASUREMENT")])
def test_load_config_refuses_an_inconsistent_energy_score_neuralgcm_loss(
        tmp_path, edit, match):
    config = copy.deepcopy(driver.load_config(_with_table(EXP10, tmp_path)))
    edit(config)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=match):
        driver.load_config(path)


def _deterministic_network(lid_card, lid_geometry, seed=3):
    config = driver.load_config(EXP4_SYMMETRIC)
    spec, _ = driver.network_spec(config, lid_card, lid_geometry)
    return SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(seed))


def _bits(tree_a, tree_b):
    return all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(
        jax.tree.leaves(tree_a), jax.tree.leaves(tree_b), strict=True))


def test_eq7_neuralgcm_programs_at_the_full_band_are_the_eq7_programs_bit_for_bit(
        lid_card, lid_geometry):
    network = _deterministic_network(lid_card, lid_geometry)
    initial, target = _lid_batch(lid_card)
    key = jax.random.PRNGKey(1)
    loss = {"kind": "eq7_neuralgcm", "calibration_window_days": 1,
            "lead_time_cutoff": FULL_BAND_TABLE}
    parent = _programs(lid_card, lid_geometry, {"kind": "eq7"}, None)
    zero = _programs(lid_card, lid_geometry, loss, (0.0, 0.0))
    p_value, p_grads = parent["value_and_grad"](network, initial, target, key)
    value, grads = zero["value_and_grad"](network, initial, target, key)
    assert float(value) == float(p_value) > 0.0 and _bits(grads, p_grads)
    total, eq7, spectrum, bias = (float(x) for x in zero["report"](
        network, initial, target, key))
    # (the report is another compiled program than value_and_grad: same
    # arithmetic, last-bit freedom)
    assert total == eq7 and spectrum > 0.0 and bias > 0.0
    assert eq7 == pytest.approx(float(p_value), rel=1.0e-13)
    # Weighted: the total adds the two terms; every trainable leaf is live.
    weighted = _programs(lid_card, lid_geometry, loss, (2.0, 3.0))
    value, grads = weighted["value_and_grad"](network, initial, target, key)
    assert float(value) == pytest.approx(
        eq7 + 2.0 * spectrum + 3.0 * bias, rel=1.0e-12)
    assert_network_gradients_live(grads)
    # A narrower cutoff table is a different loss (the lead filter reaches the
    # compiled program), and a missing table is refused, not defaulted.
    narrow = _programs(lid_card, lid_geometry, loss | {
        "lead_time_cutoff": LID_TABLE}, (0.0, 0.0))
    assert float(narrow["loss"](network, initial, target, key)) \
        < float(p_value)
    with pytest.raises(ValueError, match="PENDING_TWIN_MEASUREMENT"):
        _programs(lid_card, lid_geometry,
                  loss | {"lead_time_cutoff": PENDING}, (1.0, 1.0))
    with pytest.raises(ValueError, match="takes no loss constants"):
        _programs(lid_card, lid_geometry, loss, None)


def test_energy_score_neuralgcm_programs_at_the_full_band_match_the_energy_score(
        lid_card, lid_geometry):
    network = _channel(lid_card, lid_geometry, 1.0)
    initial, target = _lid_batch(lid_card)
    key = jax.random.PRNGKey(1)
    normalizers = swg_online.BalancedNormalizers(1.0e-10, 1.0e-6, 1.0e-6)
    parent = _programs(lid_card, lid_geometry,
                       {"kind": "energy_score", "members": 3}, normalizers)
    loss = {"kind": "energy_score_neuralgcm", "members": 3,
            "calibration_window_days": 1, "lead_time_cutoff": FULL_BAND_TABLE}
    zero = _programs(lid_card, lid_geometry, loss, (normalizers, 0.0))
    p_value, p_grads = parent["value_and_grad"](network, initial, target, key)
    value, grads = zero["value_and_grad"](network, initial, target, key)
    assert float(value) == float(p_value) > 0.0 and _bits(grads, p_grads)
    total, distance, spread, energy, spectrum = (float(x) for x in zero[
        "report"](network, initial, target, key))
    assert total == energy and spectrum > 0.0
    assert energy == pytest.approx(float(p_value), rel=1.0e-13)
    assert (distance, spread) == tuple(
        float(x) for x in parent["report"](network, initial, target, key)[1:])
    weighted = _programs(lid_card, lid_geometry, loss, (normalizers, 5.0))
    value, grads = weighted["value_and_grad"](network, initial, target, key)
    assert float(value) == pytest.approx(energy + 5.0 * spectrum, rel=1.0e-12)
    assert_network_gradients_live(grads)        # includes the xi weights
    narrow = _programs(lid_card, lid_geometry,
                       loss | {"lead_time_cutoff": LID_TABLE},
                       (normalizers, 0.0))
    assert float(narrow["loss"](network, initial, target, key)) \
        < float(p_value)


def test_the_bias_term_needs_the_whole_batch_in_one_call(tmp_path, small_card,
                                                         textured_record):
    for kind in ("eq7", "energy_score", "eq7_window_statistics",
                 "energy_score_neuralgcm"):
        for per_call in (1, 2, 5, 10):
            driver.require_whole_batch_for_bias(kind, per_call)
    driver.require_whole_batch_for_bias("eq7_neuralgcm", 10)
    for per_call in (1, 2, 5):
        with pytest.raises(ValueError, match="windows_per_call must be 10"):
            driver.require_whole_batch_for_bias("eq7_neuralgcm", per_call)
    config = _arm_config(
        textured_record, {"kind": "eq7_neuralgcm", "calibration_window_days": 1,
                          "lead_time_cutoff": SMALL_TABLE}, "none")
    path = _write_config(tmp_path / "config.json", config)
    with pytest.raises(ValueError, match="windows_per_call must be 10"):
        driver.train(config, path, card=small_card, seed=11,
                     run_dir=tmp_path / "run", device=CPU, windows_per_call=5,
                     commit="0" * 40)
    assert not (tmp_path / "run").exists()


SMALL_TABLE = {"lead_hours": [0, 6, 12], "cutoff_band": [1000, 8, 4]}


@pytest.fixture(scope="module")
def neuralgcm_eq7_run(tmp_path_factory, small_card, textured_record):
    folder = tmp_path_factory.mktemp("swg_neuralgcm_eq7")
    config = _arm_config(
        textured_record,
        {"kind": "eq7_neuralgcm", "calibration_window_days": 1,
         "lead_time_cutoff": SMALL_TABLE}, "none")
    path = _write_config(folder / "config.json", config)
    log = driver.train(config, path, card=small_card, seed=11,
                       run_dir=folder / "run", device=CPU,
                       windows_per_call=10, commit="0" * 40)
    return {"folder": folder, "config": config, "path": path, "log": log}


def test_eq7_neuralgcm_run_sets_both_weights_once_and_records_them(
        neuralgcm_eq7_run, small_card):
    run = neuralgcm_eq7_run["folder"] / "run"
    log = neuralgcm_eq7_run["log"]
    assert log["status"] == "complete"
    assert log["gates"]["first_step_no_inert"]["passed"]
    manifest = json.loads((run / "run_manifest.json").read_text())
    constants = manifest["loss_constants"]
    calibration = constants["calibration"]
    # Each added term equals the filtered Eq. 7 term on the first batch.
    assert constants["spectrum_weight"] * calibration["spectrum_mse"] \
        == pytest.approx(calibration["eq7_filtered"], rel=1.0e-12)
    assert constants["bias_weight"] * calibration["bias_mse"] \
        == pytest.approx(calibration["eq7_filtered"], rel=1.0e-12)
    # ... recomputed independently on the contract's first batch.
    config = neuralgcm_eq7_run["config"]
    contract = driver.SWGDatasetContract.from_json(
        run / "dataset_contract.json", card=small_card)
    batch = next(contract.batches(split="train", window_days=1, epoch=0,
                                  seed=11, target_interval_hours=1))
    assert calibration["start_absolute_hours"] == [
        int(hour) for hour in batch.start_absolute_hour]
    geometry = build_closure_geometry(small_card)
    spec, _ = driver.network_spec(config, small_card, geometry)
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(11))
    programs = driver.build_batch_programs(
        small_card, geometry, window_days=1, target_interval_hours=1,
        objective=config["objective"], wall_ring=config["wall_ring"],
        training_precision="fp64", checkpoint=True, loss=config["loss"],
        constants=(0.0, 0.0))
    _, eq7, spectrum, bias = driver.batch_report(
        programs["report"], network, batch, jax.random.PRNGKey(11), 10)
    assert calibration["eq7_filtered"] == pytest.approx(eq7, rel=1.0e-9)
    assert calibration["spectrum_mse"] == pytest.approx(spectrum, rel=1.0e-9)
    assert calibration["bias_mse"] == pytest.approx(bias, rel=1.0e-9)
    # Frozen: every logged epoch's total is built from those same weights.
    for entry in log["epochs"]:
        report = entry["validation_report"]
        assert report["total"] == pytest.approx(
            report["eq7_filtered"]
            + constants["spectrum_weight"] * report["spectrum_mse"]
            + constants["bias_weight"] * report["bias_mse"], rel=1.0e-9)
        assert report["total"] == pytest.approx(
            entry["validation_loss"], rel=1.0e-12)
        assert np.isfinite(entry["train_loss_mean"])
    assert manifest["loss"]["lead_time_cutoff"] == SMALL_TABLE
    assert manifest["loss"]["weights"].startswith("both set once")


def test_eq7_neuralgcm_run_resumes_with_the_recorded_weights_or_refuses(
        neuralgcm_eq7_run, small_card, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(neuralgcm_eq7_run["folder"] / "run", run)
    before = json.loads((run / "run_manifest.json").read_text())
    driver.train(
        neuralgcm_eq7_run["config"], neuralgcm_eq7_run["path"], card=small_card,
        seed=11, run_dir=run, device=CPU, windows_per_call=10, commit="0" * 40)
    assert json.loads((run / "run_manifest.json").read_text()) == before
    stripped = {key: value for key, value in before.items()
                if key != "loss_constants"}
    (run / "run_manifest.json").write_text(json.dumps(stripped))
    with pytest.raises(RuntimeError, match="records no loss constants"):
        driver.frozen_loss_constants(
            neuralgcm_eq7_run["config"], run / "run_manifest.json",
            have_epochs=True)
    wrong = before | {"loss_constants": {"kind": "eq7_window_statistics"}}
    (run / "run_manifest.json").write_text(json.dumps(wrong))
    with pytest.raises(RuntimeError, match="loss constants are for"):
        driver.frozen_loss_constants(
            neuralgcm_eq7_run["config"], run / "run_manifest.json",
            have_epochs=True)
    # A changed cutoff table is a changed run: the stored identity differs.
    changed = copy.deepcopy(neuralgcm_eq7_run["config"])
    changed["loss"]["lead_time_cutoff"] = {
        "lead_hours": [0, 6, 12], "cutoff_band": [1000, 8, 3]}
    path = _write_config(tmp_path / "changed.json", changed)
    (run / "run_manifest.json").write_text(json.dumps(before))
    with pytest.raises(RuntimeError, match="refusing to resume"):
        driver.train(changed, path, card=small_card, seed=11, run_dir=run,
                     device=CPU, windows_per_call=10, commit="0" * 40)


@pytest.fixture(scope="module")
def neuralgcm_energy_run(tmp_path_factory, small_card, textured_record,
                         trained):
    folder = tmp_path_factory.mktemp("swg_neuralgcm_energy")
    config = _arm_config(
        textured_record,
        {"kind": "energy_score_neuralgcm", "members": 2,
         "calibration_window_days": 1, "lead_time_cutoff": SMALL_TABLE},
        "white_input")
    config["network"]["initial_checkpoint"] = str(
        trained["folder"] / "run" / "checkpoints" / "w01_e02.eqx")
    config["network"]["initial_acceleration_gate"] = "recorded_only"
    path = _write_config(folder / "config.json", config)
    log = driver.train(config, path, card=small_card, seed=ENERGY_SEED,
                       run_dir=folder / "run", device=CPU,
                       windows_per_call=10, commit="0" * 40)
    return {"folder": folder, "config": config, "path": path, "log": log}


def test_energy_score_neuralgcm_run_freezes_normalizers_and_the_spectrum_weight(
        neuralgcm_energy_run, energy_run, small_card):
    run = neuralgcm_energy_run["folder"] / "run"
    log = neuralgcm_energy_run["log"]
    assert log["status"] == "complete"
    assert log["gates"]["first_step_no_inert"]["passed"]   # xi weights too
    manifest = json.loads((run / "run_manifest.json").read_text())
    constants = manifest["loss_constants"]
    assert constants["kind"] == "energy_score_neuralgcm"
    # The normalizers are the energy-score arm's: same machinery, same data.
    arm_b = json.loads((energy_run["folder"] / "run" / "run_manifest.json"
                        ).read_text())["loss_constants"]
    assert constants["normalizers"] == arm_b["normalizers"]
    calibration = constants["calibration"]
    assert constants["spectrum_weight"] * calibration["spectrum_mse"] \
        == pytest.approx(calibration["energy_score_filtered"], rel=1.0e-12)
    # ... recomputed independently: the first batch, the first training
    # step's noise key, the starting network (the loaded file plus xi).
    config = neuralgcm_energy_run["config"]
    contract = driver.SWGDatasetContract.from_json(
        run / "dataset_contract.json", card=small_card)
    batch = next(contract.batches(split="train", window_days=1, epoch=0,
                                  seed=ENERGY_SEED, target_interval_hours=1))
    assert calibration["start_absolute_hours"] == [
        int(hour) for hour in batch.start_absolute_hour]
    geometry = build_closure_geometry(small_card)
    spec, _ = driver.network_spec(config, small_card, geometry)
    optimizer = create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))
    key = jax.random.PRNGKey(ENERGY_SEED)
    network, _ = driver.initial_network(config, spec, optimizer, key=key)
    step_key = key
    for part in (driver.TRAIN_NOISE_TAG, 1, 1, 0):
        step_key = jax.random.fold_in(step_key, part)
    programs = driver.build_batch_programs(
        small_card, geometry, window_days=1, target_interval_hours=1,
        objective=config["objective"], wall_ring=config["wall_ring"],
        training_precision="fp64", checkpoint=True, loss=config["loss"],
        constants=(swg_online.BalancedNormalizers(**constants["normalizers"]),
                   0.0))
    report = driver.batch_report(programs["report"], network, batch, step_key,
                                 10, per_part_keys=True)
    assert calibration["energy_score_filtered"] == pytest.approx(
        report[3], rel=1.0e-9)
    assert calibration["spectrum_mse"] == pytest.approx(report[4], rel=1.0e-9)
    for entry in log["epochs"]:
        report = entry["validation_report"]
        assert report["total"] == pytest.approx(
            report["energy_score_filtered"]
            + constants["spectrum_weight"] * report["spectrum_mse"],
            rel=1.0e-9)
        assert report["total"] == pytest.approx(
            entry["validation_loss"], rel=1.0e-12)
        assert report["member_spread"] > 0.0
        assert report["spread_over_member_truth_distance"] == pytest.approx(
            report["member_spread"] / report["member_truth_distance"])
    assert manifest["loss"]["members"] == 2
    assert manifest["loss"]["lead_time_cutoff"] == SMALL_TABLE
    assert manifest["network"]["spec"]["noise_channel"] is True
