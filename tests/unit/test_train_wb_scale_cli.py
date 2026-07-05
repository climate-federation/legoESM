"""Arg-parse + config tests for the WB scale-training entry (JAX-free, login-safe)."""
import importlib.util
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_ENTRY = _ROOT / "scripts" / "run" / "train_weatherbench_scale.py"
_spec = importlib.util.spec_from_file_location("train_wb_scale", _ENTRY)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)          # top-level must NOT import jax


def test_argparse_roundtrip():
    cfg = mod.build_scale_config_from_args(
        ["--mode", "neural_gcm", "--resolution", "0.7", "--epochs", "20",
         "--multi-step-hours", "6,12", "--eval-wb2"])
    assert cfg.mode == "neural_gcm"
    assert cfg.resolution_deg == 0.7
    assert cfg.n_epochs == 20
    assert cfg.multi_step_hours == (6, 12)
    assert cfg.eval_wb2 is True


def test_argparse_defaults_and_all_modes():
    cfg = mod.build_scale_config_from_args([])
    assert cfg.mode == "neural_gcm" and cfg.resolution_deg == 0.7 and cfg.grad_accum == 1
    for m in ("physics", "neural_gcm", "sfno"):
        assert mod.build_scale_config_from_args(["--mode", m]).mode == m


def test_argparse_rejects_bad_mode():
    with pytest.raises(SystemExit):
        mod.build_scale_config_from_args(["--mode", "bogus"])


def test_config_yaml_loads():
    y = yaml.safe_load(open(_ROOT / "config" / "wb" / "scale" / "train_07deg.yaml"))
    assert y["grid"] == "latlon" and y["dt"] > 0
    assert {"w_T", "multi_step_hours"} <= set(y["loss"])
    assert y["loss"]["w_spec_crps_T"] == 0.0            # spectral-CRPS off at scale
    assert y["train_years"] and y["eval_years"] == [2020]
