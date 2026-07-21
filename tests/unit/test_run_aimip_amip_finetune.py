"""Smoke test for the AMIP stability fine-tune driver: importability + the
pure config-merge helper. The training loop is a GPU integration run.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_DRIVER = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "run" / "run_aimip_amip_finetune.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_amip_ft", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_merged_cfg_applies_overlay(tmp_path):
    mod = _load()
    base = tmp_path / "base.yaml"
    base.write_text("n_max: 63\nn_levels: 8\ndt: 600.0\naimip_convection: tiedtke\n")
    (tmp_path / "variant_classical.yaml").write_text("aimip_convection: edmf\n")
    suite = tmp_path / "suite.yaml"
    suite.write_text(f"base: {base}\ncfg_overrides:\n  aimip_lr: 0.001\n")
    cfg = mod._merged_cfg(suite)
    assert cfg["aimip_convection"] == "edmf"   # overlay wins
    assert cfg["nlev"] == 8                      # n_levels -> nlev
    assert cfg["aimip_lr"] == 0.001


def test_spectral_amip_rollout_has_use_checkpoint():
    import inspect
    spec = importlib.util.find_spec("legoesm.training.neural_gcm_spectral")
    if spec is None:
        pytest.skip("legoesm not importable")
    mod = importlib.import_module("legoesm.training.neural_gcm_spectral")
    sig = inspect.signature(mod.spectral_amip_rollout)
    assert "use_checkpoint" in sig.parameters, "fine-tune needs the AD-checkpoint flag"


def test_merged_cfg_selects_variant_overlay(tmp_path):
    """--variant column_nn/sfno_physics fine-tunes must merge THAT variant's
    overlay (nn/sfno hyperparams define the checkpoint architecture)."""
    mod = _load()
    base = tmp_path / "base.yaml"
    base.write_text("n_max: 63\nn_levels: 8\ndt: 600.0\n")
    (tmp_path / "variant_classical.yaml").write_text("aimip_convection: edmf\n")
    (tmp_path / "variant_column_nn.yaml").write_text("nn_hidden_dim: 123\n")
    (tmp_path / "variant_sfno_physics.yaml").write_text("sfno_embed_dim: 48\n")
    suite = tmp_path / "suite.yaml"
    suite.write_text(f"base: {base}\n")
    assert mod._merged_cfg(suite, "column_nn")["nn_hidden_dim"] == 123
    assert mod._merged_cfg(suite, "sfno_physics")["sfno_embed_dim"] == 48
    assert mod._merged_cfg(suite, "classical")["aimip_convection"] == "edmf"
