"""Arg-parse tests for the WB2 checkpoint eval driver (JAX-free, login-safe).

Mirrors ``test_train_wb_scale_cli.py``: the CLI module's top level must NOT
import jax, so it loads on a login node and this test exercises only the
argparse / config layer.
"""
import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_ENTRY = _ROOT / "scripts" / "validate" / "run_weatherbench_eval.py"
_spec = importlib.util.spec_from_file_location("run_wb_eval", _ENTRY)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)          # top-level must NOT import jax


def test_argparse_roundtrip():
    cfg = mod.build_eval_config_from_args(
        ["--config", "cfg.yaml", "--mode", "neural_gcm", "--checkpoint",
         "ck/epoch_0003.eqx", "--leads", "24,72", "--eval-year", "2020",
         "--n-inits", "3", "--init-stride-hours", "48", "--resolution-deg", "1.5",
         "--out", "out/sc.json"])
    assert cfg.mode == "neural_gcm"
    assert cfg.config_path == "cfg.yaml"
    assert cfg.training_core == "spectral"          # default
    assert cfg.checkpoint == "ck/epoch_0003.eqx"
    assert cfg.leads_hours == (24, 72)
    assert cfg.eval_year == 2020
    assert cfg.n_inits == 3
    assert cfg.init_stride_hours == 48
    assert cfg.resolution_deg == 1.5
    assert cfg.out == "out/sc.json"


def test_argparse_defaults():
    cfg = mod.build_eval_config_from_args(
        ["--config", "c.yaml", "--checkpoint", "e.eqx"])
    assert cfg.mode == "neural_gcm"
    assert cfg.training_core == "spectral"
    assert cfg.leads_hours == (24, 72)              # default leads
    assert cfg.eval_year is None                    # -> YAML eval_years[0] in main
    assert cfg.n_inits == 4 and cfg.init_stride_hours == 24
    assert cfg.resolution_deg == 1.5


def test_all_modes_accepted():
    for m in ("physics", "neural_gcm", "sfno"):
        cfg = mod.build_eval_config_from_args(
            ["--config", "c.yaml", "--checkpoint", "e.eqx", "--mode", m])
        assert cfg.mode == m


def test_rejects_bad_mode():
    with pytest.raises(SystemExit):
        mod.build_eval_config_from_args(
            ["--config", "c.yaml", "--checkpoint", "e.eqx", "--mode", "bogus"])


def test_latlon_core_hard_errors():
    """v1 is spectral-only: --training-core latlon is a hard SystemExit."""
    with pytest.raises(SystemExit, match="latlon is not supported"):
        mod.build_eval_config_from_args(
            ["--config", "c.yaml", "--checkpoint", "e.eqx",
             "--training-core", "latlon"])


def test_missing_required_args_exit():
    with pytest.raises(SystemExit):
        mod.build_eval_config_from_args([])                     # no --config/--checkpoint
    with pytest.raises(SystemExit):
        mod.build_eval_config_from_args(["--config", "c.yaml"])  # no --checkpoint


def test_bad_leads_rejected():
    with pytest.raises(SystemExit):
        mod.build_eval_config_from_args(
            ["--config", "c.yaml", "--checkpoint", "e.eqx", "--leads", "0,24"])
    with pytest.raises(SystemExit):
        mod.build_eval_config_from_args(
            ["--config", "c.yaml", "--checkpoint", "e.eqx", "--leads", ""])


def test_assert_leads_on_dt_grid():
    """The dt-grid guard: spectral dt=450 s divides every hour-lead (3600/450=8);
    a dt that does NOT divide the lead's seconds hard-errors."""
    mod._assert_leads_on_dt_grid((6, 24, 72), 450.0)             # 8, 192, 576 steps -> ok
    with pytest.raises(SystemExit, match="not an exact multiple"):
        mod._assert_leads_on_dt_grid((1,), 700.0)               # 3600/700 = 5.14 -> reject


def test_nest_scorecard():
    flat = {("z500", 24): {"rmse": 1.0, "acc": 0.9, "bias": 0.1},
            ("z500", 72): {"rmse": 2.0, "acc": 0.8, "bias": 0.2},
            ("t850", 24): {"rmse": 0.5, "acc": 0.95, "bias": 0.0}}
    nested = mod._nest_scorecard(flat)
    assert nested["z500"]["24"]["rmse"] == 1.0
    assert nested["z500"]["72"]["acc"] == 0.8
    assert set(nested["t850"].keys()) == {"24"}
