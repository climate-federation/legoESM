"""Component/dycore registry — discovery, validation, plugin entry points (B4)."""

from __future__ import annotations

import pytest

from legoesm.components import DycoreProtocol
from legoesm.registry import DYCORE_REGISTRY, Registry, create_dycore


class _ToyDycore:
    def __init__(self, grid, config=None):
        self.grid = grid

    def step(self, state, dt):
        return state


class _BadDycore:
    def __init__(self, grid, config=None):
        pass

    step = 1  # passes isinstance(DycoreProtocol) but is not callable


def test_register_and_resolve() -> None:
    reg = Registry("dycore")
    reg.register("toy", _ToyDycore)
    assert reg.get("toy") is _ToyDycore
    assert reg.available() == ["toy"]


def test_unknown_raises_with_available() -> None:
    reg = Registry("dycore")
    reg.register("a", _ToyDycore)
    with pytest.raises(ValueError, match="Unknown dycore 'nope'.*Available: a"):
        reg.get("nope")


def test_duplicate_register_raises_unless_overwrite() -> None:
    reg = Registry("dycore")
    reg.register("toy", _ToyDycore)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("toy", _ToyDycore)
    reg.register("toy", _BadDycore, overwrite=True)  # allowed
    assert reg.get("toy") is _BadDycore


def test_create_dycore_resolves_validates_and_conforms() -> None:
    DYCORE_REGISTRY.register("toy_create", _ToyDycore, overwrite=True)
    dycore = create_dycore("toy_create", grid=object())
    assert isinstance(dycore, DycoreProtocol)


def test_create_dycore_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown dycore"):
        create_dycore("definitely_not_registered_xyz", grid=object())


def test_create_dycore_rejects_malformed_plugin() -> None:
    """A factory whose product has a non-callable step is caught at resolution."""
    DYCORE_REGISTRY.register("toy_bad", _BadDycore, overwrite=True)
    with pytest.raises(TypeError, match="callable 'step'"):
        create_dycore("toy_bad", grid=object())


def test_entry_point_plugin_discovery(monkeypatch) -> None:
    """A plugin advertised via an entry point resolves with no in-process register."""
    import legoesm.registry as reg_mod

    class _EP:
        name = "plugin_dycore"

        def load(self):
            return _ToyDycore

    def _fake_entry_points(*, group):
        assert group == "legoesm.dycores"
        return [_EP()]

    monkeypatch.setattr(reg_mod, "entry_points", _fake_entry_points, raising=False)
    # Patch the importlib.metadata name the method imports locally.
    import importlib.metadata as md

    monkeypatch.setattr(md, "entry_points", _fake_entry_points)

    reg = Registry("dycore")
    assert reg.get("plugin_dycore") is _ToyDycore  # discovered via entry point
