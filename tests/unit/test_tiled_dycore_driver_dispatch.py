"""Decision-logic of ``ModelDriver._maybe_build_tiled_step`` (P4 increment
1b) — the flag/layout/experimental refusal ladder and the EXACT mirror of
``build_segment_fn``'s inner dynamics-copy predicate (codex round-14 High +
Medium).  Tested via a lightweight stub (the helper reads only
``self.config``, ``self._device_config``, ``self.model``) so no full
ModelDriver construction is needed.  The tiled numerics themselves are
covered by tests/parallel/test_tiled_cc_step_adapter.py."""
from __future__ import annotations

import pytest

from legoesm.driver.config import ExperimentConfig
from legoesm.driver.model_driver import ModelDriver
from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
)

_ENV = "LEGOESM_TILED_DYCORE_EXPERIMENTAL"


class _DeviceConfig:
    def __init__(self, tiling=(2, 2), mesh="mesh-sentinel"):
        self.tiling = tiling
        self.mesh = mesh


class _Model:
    def __init__(self, config):
        self.config = config


class _Stub:
    """Minimal stand-in exposing only what the helper reads."""

    _maybe_build_tiled_step = ModelDriver._maybe_build_tiled_step

    def __init__(self, cfg, device_config, model_cfg):
        self.config = cfg
        self._device_config = device_config
        self.model = _Model(model_cfg)


def _cfg(**dycore_over):
    cfg = ExperimentConfig()
    return cfg._replace(
        enable_tiled_dycore=True,
        grid=cfg.grid._replace(grid_type="cubed_sphere"),
        dycore=cfg.dycore._replace(**dycore_over),
    )


def _model_cfg(**over):
    return CDGridPrimitiveEquationConfig(**over)


def test_flag_off_returns_none():
    stub = _Stub(_cfg()._replace(enable_tiled_dycore=False),
                 _DeviceConfig(), _model_cfg())
    assert stub._maybe_build_tiled_step(600.0) is None


@pytest.mark.parametrize("dc", [
    None,
    _DeviceConfig(mesh=None),
    _DeviceConfig(tiling=(1, 1)),
])
def test_flag_on_without_tiled_layout_is_loud(dc):
    """Flag-on with no sub-face-tiled layout must raise, never silently
    fall back to the untiled path."""
    stub = _Stub(_cfg(), dc, _model_cfg())
    with pytest.raises(ValueError, match="sub-face-tiled"):
        stub._maybe_build_tiled_step(600.0)


def test_non_square_tiling_is_loud():
    stub = _Stub(_cfg(), _DeviceConfig(tiling=(2, 3)), _model_cfg())
    with pytest.raises(ValueError, match="square"):
        stub._maybe_build_tiled_step(600.0)


def test_refuses_without_experimental_env(monkeypatch):
    """The outer compiled-segment sharding composition is not yet
    device-validated (segment runs device_config=None under tiling —
    codex round-14 High): refuse without the explicit opt-in env."""
    monkeypatch.delenv(_ENV, raising=False)
    stub = _Stub(_cfg(), _DeviceConfig(), _model_cfg())
    with pytest.raises(NotImplementedError, match=_ENV):
        stub._maybe_build_tiled_step(600.0)


def test_refuses_effective_zero_mean_active(monkeypatch):
    """Outer fixer OFF + inner zero_mean_ps_tendency active (the dycore
    gate ``zm and not (ucf and fm)``): the untiled inner model applies a
    per-RK-stage term the tiled base cut omits — refuse, never silently
    change numerics (codex round-14 Medium)."""
    monkeypatch.setenv(_ENV, "1")
    stub = _Stub(
        _cfg(fix_mass=False),
        _DeviceConfig(),
        _model_cfg(use_conservation_fixer=False, fix_mass=False,
                   zero_mean_ps_tendency=True),
    )
    with pytest.raises(NotImplementedError, match="zero_mean_ps_tendency"):
        stub._maybe_build_tiled_step(600.0)


def _capture_adapter(monkeypatch):
    """Intercept make_tiled_cc_step; return the captured-call dict."""
    import legoesm.atmosphere.dynamics.tiled_step_adapter as _ad
    seen = {}

    def _fake(model, mesh, kt, dt):
        seen.update(model=model, mesh=mesh, kt=kt, dt=dt)
        return "tiled-step-sentinel"

    monkeypatch.setattr(_ad, "make_tiled_cc_step", _fake)
    return seen


def test_mirror_disables_inner_fixer_and_zero_mean(monkeypatch):
    """Outer fix_mass=True AND model fix_mass=True: the copy handed to the
    adapter must have fix_mass=False + zero_mean_ps_tendency=False —
    exactly build_segment_fn's inner dynamics copy."""
    monkeypatch.setenv(_ENV, "1")
    seen = _capture_adapter(monkeypatch)
    model_cfg = _model_cfg(use_conservation_fixer=True, fix_mass=True,
                           zero_mean_ps_tendency=True)
    stub = _Stub(_cfg(fix_mass=True), _DeviceConfig(), model_cfg)
    out = stub._maybe_build_tiled_step(600.0)
    assert out == "tiled-step-sentinel"
    assert seen["model"].config.fix_mass is False
    assert seen["model"].config.zero_mean_ps_tendency is False
    assert seen["kt"] == 2 and seen["dt"] == 600.0
    assert seen["mesh"] == "mesh-sentinel"
    # The driver's own model is NEVER mutated (shallow copy only).
    assert stub.model.config.fix_mass is True
    assert stub.model.config.zero_mean_ps_tendency is True


def test_mirror_predicate_requires_model_fix_mass(monkeypatch):
    """Outer fix_mass=True but model fix_mass=False: build_segment_fn does
    NOT copy in this case, so neither may the helper — the config reaches
    the adapter unmutated (zm already False here to stay in-envelope)."""
    monkeypatch.setenv(_ENV, "1")
    seen = _capture_adapter(monkeypatch)
    model_cfg = _model_cfg(use_conservation_fixer=False, fix_mass=False,
                           zero_mean_ps_tendency=False)
    stub = _Stub(_cfg(fix_mass=True), _DeviceConfig(), model_cfg)
    assert stub._maybe_build_tiled_step(600.0) == "tiled-step-sentinel"
    assert seen["model"].config is model_cfg
