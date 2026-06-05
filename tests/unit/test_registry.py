"""Component/dycore registry — discovery, validation, plugin entry points (B4)."""

from __future__ import annotations

import pytest

from legoesm.components import DycoreProtocol
from legoesm.dycore_factory import create_dycore
from legoesm.registry import DYCORE_REGISTRY, Registry


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
    with pytest.raises(ValueError, match="Unknown"):
        create_dycore("definitely_not_registered_xyz", grid=object())


def test_create_dycore_propagates_plugin_load_failure(monkeypatch) -> None:
    """A broken plugin override propagates — it is NOT silently replaced by the
    built-in (a ValueError from ep.load() is not a registry miss)."""
    import importlib.metadata as md

    import legoesm.registry as reg

    class _BrokenEP:
        name = "cdgrid_shallow_water"  # a built-in name, shadowed by a broken plugin

        def load(self):
            raise ValueError("plugin broken on load")

    monkeypatch.setattr(md, "entry_points", lambda *, group: [_BrokenEP()])
    # Fresh, isolated discovery state (restored after the test).
    monkeypatch.setattr(reg.DYCORE_REGISTRY, "_loaded_entry_points", False)
    monkeypatch.setattr(reg.DYCORE_REGISTRY, "_failed_entry_points", {})

    # BOTH the first and a subsequent call must propagate — a broken override is
    # never silently masked by the built-in on a later call.
    for _ in range(2):
        with pytest.raises(ValueError, match="plugin broken"):
            # create_dycore uses the same DYCORE_REGISTRY instance monkeypatched
            # via ``reg`` above (imported by reference), so the patch still applies.
            create_dycore("cdgrid_shallow_water", object())


def test_create_dycore_resolves_builtin_atmosphere_solver() -> None:
    """A built-in atmosphere dycore resolves via the fallback and validates."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere

    grid = create_cubed_sphere(8)
    dycore = create_dycore("cdgrid_shallow_water", grid)
    assert isinstance(dycore, DycoreProtocol)
    assert hasattr(dycore, "step")


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


def test_sw_barotropic_provider_resolves_in_fresh_process_without_atmosphere():
    """Substrate-owned bootstrap: in a FRESH interpreter with the atmosphere NOT
    preloaded, the 3-D ocean's shallow-water barotropic provider resolves via
    entry-point discovery (importlib.metadata) — so a cube ocean builds without
    any caller import-order dependence and without the ocean importing atmosphere.
    """
    import os
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from legoesm.registry import SW_BAROTROPIC_REGISTRY\n"
        "assert 'legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid' "
        "not in sys.modules, 'atmosphere preloaded'\n"
        "b = SW_BAROTROPIC_REGISTRY.get('fv3sw')\n"
        "assert callable(b)\n"
        "assert SW_BAROTROPIC_REGISTRY.get('fv3edge') is not None\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
