"""Unit tests for named loss presets (training/loss_presets.py, D5)."""

import ast
from pathlib import Path

import pytest

from legoesm.training.loss_presets import (
    available_presets,
    load_loss_preset,
    merge_loss_preset,
)

_REPO = Path(__file__).resolve().parents[2]


def _loss_config_fields() -> set:
    """LossConfig field names via AST (no jax import needed)."""
    src = (_REPO / "packages/ml/legoesm/training/losses.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "LossConfig":
            return {
                st.target.id
                for st in node.body
                if isinstance(st, ast.AnnAssign)
                and isinstance(st.target, ast.Name)
            }
    raise AssertionError("LossConfig not found")


def test_bundled_presets_exist():
    names = available_presets()
    assert "ace2" in names
    assert "ace2_flux" in names
    assert "neuralgcm" in names


@pytest.mark.parametrize("name", ["ace2", "ace2_flux", "neuralgcm"])
def test_every_preset_key_is_a_lossconfig_field(name):
    fields = _loss_config_fields()
    preset = load_loss_preset(name)
    assert preset, f"preset {name} is empty"
    unknown = set(preset) - fields
    assert not unknown, f"preset {name} has non-LossConfig keys: {sorted(unknown)}"


def test_ace2_preset_matches_validated_campaign_numbers():
    p = load_loss_preset("ace2")
    assert p["residual_normalize"] is True
    assert p["normalize_by_scale"] is False
    assert p["multi_step_hours"] == [6, 12]
    assert p["multi_step_weights"] == [1.0, 1.0]
    assert (p["w_T"], p["w_u"], p["w_v"], p["w_q"], p["w_ps"]) == (
        0.5,
        0.5,
        0.5,
        0.5,
        0.3,
    )
    assert p["level_weighting"] == "pressure"
    # pure MSE: no flux/bias/CRPS keys in the state-only preset
    assert not any(k.startswith(("w_flux", "w_bias", "w_crps")) for k in p)


def test_ace2_flux_extends_state_preset():
    state, flux = load_loss_preset("ace2"), load_loss_preset("ace2_flux")
    for k, v in state.items():
        assert flux[k] == v, f"ace2_flux diverges from ace2 on state key {k}"
    assert flux["w_flux_rsut"] == 1.0
    assert flux["w_bias_flux_olr"] == 1.0


def test_unknown_preset_raises_listing_available():
    with pytest.raises(ValueError, match="available"):
        load_loss_preset("nope_not_a_preset")


def test_explicit_path_load(tmp_path):
    f = tmp_path / "custom.yaml"
    f.write_text("loss:\n  w_T: 2.0\n")
    assert load_loss_preset(str(f)) == {"w_T": 2.0}
    with pytest.raises(ValueError, match="not found"):
        load_loss_preset(str(tmp_path / "missing.yaml"))


def test_merge_suite_wins():
    merged = merge_loss_preset({"w_T": 0.5, "w_q": 0.5}, {"w_T": 9.0})
    assert merged == {"w_T": 9.0, "w_q": 0.5}
    assert merge_loss_preset({"w_T": 0.5}, None) == {"w_T": 0.5}
