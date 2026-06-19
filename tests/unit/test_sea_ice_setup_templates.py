"""Tests for the sea-ice ``setup:`` adapter + idealized-case templates (#388).

Sea-ice is setup-only (no standalone recipe path): a ``setup: {name, grid}``
template routes through ``SeaIceExperimentConfig`` to
``run_sea_ice_test_matrix.py --test <case> --grid <grid>`` (the sea-ice
``--test`` filter is already exact), gated by ``validate_templates.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from legoesm.ice.experiment_config import SeaIceExperimentConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATES = _REPO_ROOT / "config" / "templates" / "sea_ice"


def _template_paths() -> list[Path]:
    return sorted(_TEMPLATES.glob("*.yaml"))


def _matrix_case_pairs() -> set[tuple[str, str]]:
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "matrix"))
    import run_sea_ice_test_matrix as M  # noqa: PLC0415
    cases = (M._build_test_matrix() if hasattr(M, "_build_test_matrix")
             else M.build_test_matrix())
    return {(c.case, c.grid_type) for c in cases}


class TestSeaIceTemplates:
    def test_templates_exist(self):
        assert _template_paths(), f"no sea-ice setup templates under {_TEMPLATES}"

    @pytest.mark.parametrize("path", _template_paths(), ids=lambda p: p.name)
    def test_validates_and_runs(self, path):
        cfg = SeaIceExperimentConfig.from_yaml(str(path))
        assert cfg.get("model.type") in ("sea_ice", "sea_ice_only")
        name, grid = cfg.get("setup.name"), cfg.get("setup.grid")
        cfg.validate_strict()
        cmd = cfg.run_command(str(path))
        assert "run_sea_ice_test_matrix.py" in cmd
        assert f"--test {name}" in cmd            # --test is already exact (no =)
        assert f"=" + str(name) not in cmd        # no `=` prefix for sea-ice
        assert f"--grid {grid}" in cmd

    def test_shipped_templates_select_a_real_matrix_case(self):
        pairs = _matrix_case_pairs()
        for path in _template_paths():
            cfg = SeaIceExperimentConfig.from_yaml(str(path))
            name, grid = cfg.get("setup.name"), cfg.get("setup.grid")
            assert (name, grid) in pairs, (
                f"{path.name}: ({name!r}, {grid!r}) is not a sea-ice matrix case")


class TestSeaIceAdapter:
    def _cfg(self, setup, **extra):
        d = {"model": {"type": "sea_ice_only"}}
        if setup is not None:
            d["setup"] = setup
        d.update(extra)
        return SeaIceExperimentConfig.from_dict(d)

    def test_setup_required(self):
        with pytest.raises(ValueError, match="setup-only"):
            self._cfg(None).validate_strict()

    def test_valid(self):
        self._cfg({"name": "stefan_growth", "grid": "column"}).validate_strict()

    def test_bad_grid(self):
        with pytest.raises(ValueError, match="not a valid matrix grid"):
            self._cfg({"name": "stefan_growth", "grid": "latlon"}).validate_strict()

    def test_unsupported_override(self):
        # sea-ice matrix has no --levels/--dt/--days/--resolution
        with pytest.raises(ValueError, match="not supported by"):
            self._cfg({"name": "stefan_growth", "grid": "column",
                       "dt_seconds": 30.0}).validate_strict()

    def test_ignored_section_rejected(self):
        with pytest.raises(ValueError, match="ignored by a sea-ice"):
            self._cfg({"name": "stefan_growth", "grid": "column"},
                      ice={"some_knob": 1}).validate_strict()

    def test_non_mapping_output_rejected(self):
        with pytest.raises(ValueError, match="output: must be a mapping"):
            self._cfg({"name": "stefan_growth", "grid": "column"},
                      output="out").validate_strict()

    def test_run_command_no_prefix(self):
        cmd = self._cfg({"name": "stefan_growth", "grid": "column",
                         "quick": True}).run_command()
        assert "--test stefan_growth" in cmd and "--grid column" in cmd
        assert "--quick" in cmd
        assert "=stefan_growth" not in cmd
