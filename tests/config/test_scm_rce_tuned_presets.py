"""The committed SCM-RCE tuned-parameter presets must load and stay in bounds.

A preset that names a non-existent field, or a value outside its
``__param_spec__`` bounds, would be a hard error at run time on every driver
that consumes it — better caught here.  This also pins that the presets are
keyed by the qualified ``scheme_key.field`` the ``--params`` loader expects.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_PRESET_DIR = (Path(__file__).resolve().parents[2]
               / "config" / "params" / "scm_rce_tuned_f0")


def _presets():
    return sorted(_PRESET_DIR.glob("*.yaml"))


def test_preset_dir_exists_and_is_populated():
    assert _PRESET_DIR.is_dir(), f"missing {_PRESET_DIR}"
    assert _presets(), "no preset YAMLs found"


@pytest.mark.parametrize("path", _presets(), ids=lambda p: p.stem)
def test_preset_loads_and_is_in_spec_bounds(path):
    from legoesm.driver.run_config_yaml import load_params_config
    from legoesm.training.param_collector import build_registry

    params = load_params_config(path)
    # A scheme with no moved parameters (kuo) is allowed to be header-only.
    if not params:
        return
    by_name = {m.qualified_name: m for m in build_registry()}
    for key, val in params.items():
        assert key in by_name, (
            f"{path.name}: '{key}' is not a known __param_spec__ qualified "
            "name; --params would reject it")
        lo, hi = by_name[key].bounds
        assert float(lo) <= float(val) <= float(hi), (
            f"{path.name}: {key}={val} outside spec bounds [{lo}, {hi}]")
