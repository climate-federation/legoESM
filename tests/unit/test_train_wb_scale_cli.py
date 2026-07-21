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
    # --resolution defaults to None (#817 papercut fix): the grid comes from
    # the YAML; the value is derived for logging and an explicit mismatch is a
    # hard error (see test_resolution_yaml_check below).
    assert cfg.mode == "neural_gcm" and cfg.resolution_deg is None and cfg.grad_accum == 1
    assert cfg.training_core == "latlon"   # default core: byte-unchanged path
    for m in ("physics", "neural_gcm", "sfno"):
        assert mod.build_scale_config_from_args(["--mode", m]).mode == m


def test_argparse_rejects_bad_mode():
    with pytest.raises(SystemExit):
        mod.build_scale_config_from_args(["--mode", "bogus"])


def test_argparse_training_core_roundtrip_and_rejects_bad():
    """#817: --training-core selects the spectral (semi-implicit) training
    core; unknown values are rejected by argparse choices."""
    for core in ("latlon", "spectral"):
        cfg = mod.build_scale_config_from_args(["--training-core", core])
        assert cfg.training_core == core
    with pytest.raises(SystemExit):
        mod.build_scale_config_from_args(["--training-core", "bogus"])


def test_resolution_yaml_check():
    """#817 papercut: --resolution must MATCH the YAML grid or hard-error —
    the old flag silently logged one resolution while training at another."""
    yml = {"n_lat": 256, "n_lon": 512}
    # None (default) -> derived from the YAML.
    cfg = mod.build_scale_config_from_args([])
    assert abs(mod._check_resolution_matches_yaml(cfg, yml) - 180.0 / 256) < 1e-9
    # Matching explicit value passes.
    cfg = mod.build_scale_config_from_args(["--resolution", "0.703125"])
    assert mod._check_resolution_matches_yaml(cfg, yml) == 0.703125
    # Mismatch (2.8 deg vs a 0.7 deg YAML) -> SystemExit, not a silent no-op.
    cfg = mod.build_scale_config_from_args(["--resolution", "2.8"])
    with pytest.raises(SystemExit, match="does not match the YAML"):
        mod._check_resolution_matches_yaml(cfg, yml)


def test_config_yaml_loads():
    y = yaml.safe_load(open(_ROOT / "config" / "wb" / "scale" / "train_07deg.yaml"))
    assert y["grid"] == "latlon" and y["dt"] > 0
    assert {"w_T", "multi_step_hours"} <= set(y["loss"])
    assert y["loss"]["w_spec_crps_T"] == 0.0            # spectral-CRPS off at scale
    assert y["train_years"] and y["eval_years"] == [2020]


def test_n_days_cli_roundtrip_and_reject():
    """#1047 ask a: --n-days sets the training-window length; default None."""
    assert mod.build_scale_config_from_args([]).n_days is None
    assert mod.build_scale_config_from_args(["--n-days", "60"]).n_days == 60
    with pytest.raises(SystemExit, match="--n-days must be >= 1"):
        mod.build_scale_config_from_args(["--n-days", "0"])


def test_resolve_n_days_precedence():
    """CLI cfg.n_days > YAML n_training_days > 3; --smoke forces 1 (#1047)."""
    from legoesm.training.scale_build import _resolve_n_days

    base = mod.build_scale_config_from_args([])          # n_days=None, smoke=False
    assert _resolve_n_days(base, {}) == 3                # historical default
    assert _resolve_n_days(base, {"n_training_days": 60}) == 60   # YAML wins over 3
    cli = mod.build_scale_config_from_args(["--n-days", "10"])
    assert _resolve_n_days(cli, {"n_training_days": 60}) == 10    # CLI wins over YAML
    assert _resolve_n_days(base._replace(smoke=True), {"n_training_days": 60}) == 1
    with pytest.raises(ValueError, match="must be >= 1"):
        _resolve_n_days(base, {"n_training_days": 0})


def test_rollout_hours_matches_first_lead():
    """The training rollout horizon = the FIRST multi_step_hours lead — the same
    lead load_era5_samples uses to pick the target, so pred and target stay at
    the same forecast time (the old hardwired single_day_rollout scored a 24 h
    forecast against a 6 h target)."""
    from legoesm.training.scale_build import rollout_hours

    cfg = mod.build_scale_config_from_args(["--multi-step-hours", "12,24"])
    assert rollout_hours(cfg, {}) == 12.0
    # CLI default (6,12) -> 6 h
    cfg = mod.build_scale_config_from_args([])
    assert rollout_hours(cfg, {}) == 6.0
    # no CLI leads -> YAML loss.multi_step_hours wins; nothing at all -> 6 h
    cfg = cfg._replace(multi_step_hours=())
    assert rollout_hours(cfg, {"loss": {"multi_step_hours": [12, 24]}}) == 12.0
    assert rollout_hours(cfg, {}) == 6.0
