"""Login-node/CI-safe tests for the WB campaign driver's import-light layer.

Exercises arg-parsing, the pure argv builders, checkpoint discovery, and the
dispatch-hardening guards WITHOUT importing JAX or touching a cluster (the heavy
train/eval/plot ``main`` calls live inside ``run_campaign``, which these tests do
not invoke).
"""
import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _load_campaign():
    spec = importlib.util.spec_from_file_location(
        "run_weatherbench_campaign",
        _REPO / "scripts" / "run" / "run_weatherbench_campaign.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_campaign()


def test_defaults_parse():
    cfg = C.build_campaign_config_from_args([])
    assert cfg.modes == ("physics", "neural_gcm", "sfno")
    assert cfg.stages == ("train", "eval", "plot")
    assert cfg.training_core == "spectral"
    assert cfg.leads_hours == (24, 72, 120, 240)
    assert cfg.config_path.endswith("spectral_t63.yaml")
    assert cfg.sota_csv and cfg.sota_csv.endswith(".csv")


def test_modes_subset_and_dedup():
    cfg = C.build_campaign_config_from_args(["--modes", "sfno,physics,sfno"])
    assert cfg.modes == ("sfno", "physics")   # order preserved, de-duped


def test_unknown_mode_is_hard_error():
    with pytest.raises(SystemExit):
        C.build_campaign_config_from_args(["--modes", "physics,bogus"])


def test_unknown_stage_is_hard_error():
    with pytest.raises(SystemExit):
        C.build_campaign_config_from_args(["--stages", "train,frobnicate"])


def test_empty_sota_csv_disables_it():
    cfg = C.build_campaign_config_from_args(["--sota-csv", ""])
    assert cfg.sota_csv is None


def test_bad_leads_rejected():
    with pytest.raises(SystemExit):
        C.build_campaign_config_from_args(["--leads", "24,-6"])
    with pytest.raises(SystemExit):
        C.build_campaign_config_from_args(["--leads", "24,0"])


def test_bad_scalars_rejected():
    for bad in (["--epochs", "0"], ["--n-inits", "0"], ["--init-stride-hours", "0"]):
        with pytest.raises(SystemExit):
            C.build_campaign_config_from_args(bad)


def test_build_train_argv():
    cfg = C.build_campaign_config_from_args(
        ["--out-root", "/tmp/wb", "--epochs", "3", "--smoke"])
    argv = C.build_train_argv(cfg, "neural_gcm")
    assert argv[:4] == ["--config", cfg.config_path, "--mode", "neural_gcm"]
    assert "--training-core" in argv and "spectral" in argv
    assert "--out" in argv and "/tmp/wb/neural_gcm" in argv
    assert "--resume" in argv
    assert argv[argv.index("--epochs") + 1] == "3"
    assert "--smoke" in argv


def test_build_train_argv_no_epochs_no_smoke():
    cfg = C.build_campaign_config_from_args([])
    argv = C.build_train_argv(cfg, "physics")
    assert "--epochs" not in argv   # default: use the YAML value
    assert "--smoke" not in argv


def test_build_eval_argv():
    cfg = C.build_campaign_config_from_args(
        ["--out-root", "/tmp/wb", "--leads", "24,72", "--eval-year", "2020",
         "--n-inits", "5"])
    argv = C.build_eval_argv(cfg, "sfno", "/tmp/wb/sfno/epoch_0011.eqx")
    assert argv[argv.index("--mode") + 1] == "sfno"
    assert argv[argv.index("--checkpoint") + 1] == "/tmp/wb/sfno/epoch_0011.eqx"
    assert argv[argv.index("--leads") + 1] == "24,72"
    assert argv[argv.index("--eval-year") + 1] == "2020"
    assert argv[argv.index("--n-inits") + 1] == "5"
    assert argv[argv.index("--out") + 1] == "/tmp/wb/sfno/scorecard.json"


def test_build_eval_argv_omits_year_when_unset():
    cfg = C.build_campaign_config_from_args([])
    argv = C.build_eval_argv(cfg, "physics", "/tmp/c.eqx")
    assert "--eval-year" not in argv   # default: YAML eval_years[0]


def test_build_plot_argv():
    cfg = C.build_campaign_config_from_args(["--metric", "rmse"])
    fam = {"physics": "/o/physics/scorecard.json",
           "neural_gcm": "/o/neural_gcm/scorecard.json"}
    argv = C.build_plot_argv(cfg, fam, "/o/wb_scorecard.png")
    # one --scorecard NAME=PATH per family
    scs = [argv[i + 1] for i, t in enumerate(argv) if t == "--scorecard"]
    assert "physics=/o/physics/scorecard.json" in scs
    assert "neural_gcm=/o/neural_gcm/scorecard.json" in scs
    assert argv[argv.index("--sota-csv") + 1] == cfg.sota_csv
    assert argv[argv.index("--out") + 1] == "/o/wb_scorecard.png"
    assert argv[argv.index("--metric") + 1] == "rmse"


def test_build_plot_argv_no_sota():
    cfg = C.build_campaign_config_from_args(["--sota-csv", ""])
    argv = C.build_plot_argv(cfg, {"sfno": "/o/sfno/scorecard.json"}, "/o/p.png")
    assert "--sota-csv" not in argv


def test_latest_checkpoint(tmp_path):
    d = tmp_path / "physics"
    d.mkdir()
    assert C.latest_checkpoint(str(d)) is None
    for e in (1, 5, 11, 2):
        (d / f"epoch_{e:04d}.eqx").write_text("x")
    assert C.latest_checkpoint(str(d)).endswith("epoch_0011.eqx")   # numeric max
    assert C.latest_checkpoint(str(tmp_path / "missing")) is None


def test_barrier_noop_single_process():
    # nproc<=1 must not import mpi4py / raise.
    C._barrier(1)
    C._barrier(0)


def test_paths_helpers():
    assert C.mode_out_dir("/o", "sfno") == "/o/sfno"
    assert C.scorecard_path("/o", "sfno") == "/o/sfno/scorecard.json"
