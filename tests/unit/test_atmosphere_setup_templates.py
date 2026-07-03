"""Tests for the atmosphere ``setup:`` selector + idealized-case templates (#388).

Mirrors the ocean strategy for the atmosphere component: a top-level
``setup: {name, grid}`` block in the atmosphere ``Config`` adapter routes
``init_experiment --template atmosphere/<case>`` to
``run_atmosphere_test_matrix.py --test =<case> --grid <grid>`` (exact match),
validated through the shared ``legoesm.core.setup_selector`` and gated by
``validate_templates.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from legoesm.config import Config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ATM_TEMPLATES = _REPO_ROOT / "config" / "templates" / "atmosphere"


def _template_paths() -> list[Path]:
    return sorted(_ATM_TEMPLATES.glob("*.yaml"))


def _matrix_case_pairs() -> set[tuple[str, str]]:
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "matrix"))
    import run_atmosphere_test_matrix as M  # noqa: PLC0415
    cases = (M._build_test_matrix() if hasattr(M, "_build_test_matrix")
             else M.build_test_matrix())
    return {(c.case, c.grid_type) for c in cases}


class TestAtmosphereTemplates:
    def test_templates_exist(self):
        assert _template_paths(), f"no atmosphere setup templates under {_ATM_TEMPLATES}"

    @pytest.mark.parametrize("path", _template_paths(), ids=lambda p: p.name)
    def test_validates_and_runs(self, path):
        cfg = Config.from_yaml(str(path))
        assert cfg.get("model.type") in ("atmosphere", "atmosphere_only")
        name, grid = cfg.get("setup.name"), cfg.get("setup.grid")
        cfg.validate_strict()
        cmd = cfg.run_command(str(path))
        assert "run_atmosphere_test_matrix.py" in cmd
        assert f"--test ={name}" in cmd          # EXACT match via --test
        assert f"--grid {grid}" in cmd
        assert "legoesm run" not in cmd          # not the recipe path

    def test_shipped_templates_select_a_real_matrix_case(self):
        pairs = _matrix_case_pairs()
        for path in _template_paths():
            cfg = Config.from_yaml(str(path))
            name, grid = cfg.get("setup.name"), cfg.get("setup.grid")
            assert (name, grid) in pairs, (
                f"{path.name}: ({name!r}, {grid!r}) is not an atmosphere matrix "
                f"case (--test ={name} --grid {grid} would select nothing)")


class TestAtmosphereSetupSelector:
    def _cfg(self, setup, **extra):
        d = {"model": {"type": "atmosphere_only"}, "setup": setup}
        d.update(extra)
        return Config.from_dict(d)

    def test_valid(self):
        self._cfg({"name": "held_suarez", "grid": "cubed_sphere"}).validate_strict()

    def test_bad_grid_rejected(self):
        with pytest.raises(ValueError, match="not a valid matrix grid"):
            self._cfg({"name": "held_suarez", "grid": "nope"}).validate_strict()

    def test_unsupported_override_rejected(self):
        # atmosphere matrix has no --levels / --dt
        with pytest.raises(ValueError, match="not supported by"):
            self._cfg({"name": "held_suarez", "grid": "cubed_sphere",
                       "levels": 30}).validate_strict()

    def test_atmosphere_recipe_with_setup_rejected(self):
        with pytest.raises(ValueError, match="ignored by a `setup:` template"):
            self._cfg({"name": "held_suarez", "grid": "cubed_sphere"},
                      atmosphere={"dt_seconds": 300}).validate_strict()

    def test_customised_grid_or_time_with_setup_rejected(self):
        with pytest.raises(ValueError, match=r"ignored by a `setup:` template"):
            self._cfg({"name": "held_suarez", "grid": "cubed_sphere"},
                      time={"duration_hours": 999}).validate_strict()
        with pytest.raises(ValueError, match=r"\bgrid\b"):
            self._cfg({"name": "held_suarez", "grid": "cubed_sphere"},
                      grid={"resolution": 999}).validate_strict()

    def test_customised_hardware_with_setup_rejected(self):
        # A hand-authored hardware: block is ignored on the matrix path, so it
        # is flagged. (init_experiment skips its machine-precision hardware
        # INJECTION for setup templates, so the tooling never trips this.)
        with pytest.raises(ValueError, match="ignored by a `setup:` template"):
            self._cfg({"name": "held_suarez", "grid": "cubed_sphere"},
                      hardware={"precision": {"dynamics": "float64"}}).validate_strict()

    def test_output_path_allowed_other_output_rejected(self):
        # output.path is consumed (-> --output); other output.* would be ignored
        self._cfg({"name": "held_suarez", "grid": "cubed_sphere"},
                  output={"path": "out/x/"}).validate_strict()
        with pytest.raises(ValueError, match="only output.path is used"):
            self._cfg({"name": "held_suarez", "grid": "cubed_sphere"},
                      output={"checkpoint_days": 999}).validate_strict()

    def test_run_command_exact_and_days(self):
        cmd = self._cfg({"name": "held_suarez", "grid": "latlon",
                         "duration_days": 200.0}).run_command()
        assert "--test =held_suarez" in cmd and "--grid latlon" in cmd
        assert "--days 200.0" in cmd

    def test_signature_tracks_setup_and_ignores_recipe(self):
        a = self._cfg({"name": "held_suarez", "grid": "cubed_sphere"}).signature()
        b = self._cfg({"name": "held_suarez", "grid": "latlon"}).signature()
        assert a != b
        # a no-recipe atmosphere field override is a no-op on the setup path
        c = Config.from_dict({"model": {"type": "atmosphere_only"},
                              "setup": {"name": "held_suarez", "grid": "cubed_sphere"},
                              "time": {"duration_hours": 999}}).signature()
        assert c == a
