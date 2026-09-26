"""camp.apply_tuned_preset must load a scheme's preset onto its convection sub,
and refuse a preset pointed at the wrong scheme."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PRESETS = _REPO / "config" / "params" / "scm_rce_tuned_f0"


def _camp():
    path = _REPO / "scripts" / "run" / "run_scm_rce_campaign.py"
    spec = importlib.util.spec_from_file_location("scm_rce_campaign_t", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_preset_moves_a_convection_field():
    camp = _camp()
    cfg = camp.make_physics_config(convection="dca")
    _c, _s, sub0 = camp._active_subconfig(cfg, "convection")
    out, n = camp.apply_tuned_preset(cfg, "dca", _PRESETS)
    assert n >= 1
    _c, _s, sub1 = camp._active_subconfig(out, "convection")
    # dca's single tuned param (cape_threshold) must have moved.
    assert sub1.cape_threshold != sub0.cape_threshold


def test_header_only_preset_is_a_noop():
    camp = _camp()
    cfg = camp.make_physics_config(convection="kuo")
    out, n = camp.apply_tuned_preset(cfg, "kuo", _PRESETS)
    assert n == 0 and out is cfg


def test_wrong_scheme_preset_is_rejected():
    camp = _camp()
    # a dca config fed the emanuel preset must raise, not silently do nothing.
    cfg = camp.make_physics_config(convection="dca")
    with pytest.raises((ValueError, Exception)):
        camp.apply_tuned_preset(cfg, "emanuel", _PRESETS)
