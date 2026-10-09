"""Unit tests for ``legoesm.experiment_registry`` — mode-aware adapter dispatch.

Verifies mode detection precedence, lazy adapter resolution to the atmosphere
and ocean YAML boundaries, the uniform adapter protocol, and dispatch-discipline
errors on unknown modes.
"""
from __future__ import annotations

import textwrap

import pytest

from legoesm import experiment_registry as reg
from legoesm.config import Config
from legoesm.ice.experiment_config import SeaIceExperimentConfig
from legoesm.ocean.config import OceanExperimentConfig


def test_detect_mode_precedence():
    assert reg.detect_mode({"model": {"type": "ocean_only"}}) == "ocean_only"
    assert reg.detect_mode({"mode": "ocean"}) == "ocean"
    # model.type wins over mode.
    assert reg.detect_mode(
        {"model": {"type": "atmosphere_only"}, "mode": "ocean"}
    ) == "atmosphere_only"
    # Section heuristic: ocean section, no atmosphere section.
    assert reg.detect_mode({"ocean": {"A_h": 1.0}}) == "ocean"
    # Default.
    assert reg.detect_mode({}) == "atmosphere"
    assert reg.detect_mode({"atmosphere": {}, "ocean": {}}) == "atmosphere"


def test_detect_mode_rejects_unknown_explicit():
    with pytest.raises(ValueError, match="not a known experiment mode"):
        reg.detect_mode({"model": {"type": "plasma_only"}})
    with pytest.raises(ValueError, match="not a known experiment mode"):
        reg.detect_mode({"mode": "nonsense"})


def test_get_adapter_resolves_classes():
    assert reg.get_adapter("atmosphere") is Config
    assert reg.get_adapter("atmosphere_only") is Config
    assert reg.get_adapter("ocean") is OceanExperimentConfig
    assert reg.get_adapter("ocean_only") is OceanExperimentConfig
    assert reg.get_adapter("sea_ice") is SeaIceExperimentConfig
    assert reg.get_adapter("sea_ice_only") is SeaIceExperimentConfig


def test_retired_amip_template_points_to_production_deck():
    with pytest.raises(ValueError, match="config/amip/amip_production.yaml"):
        reg.load_adapter("config/templates/coupled/amip.yaml")


def test_get_adapter_unknown_raises():
    with pytest.raises(ValueError, match="no config adapter registered"):
        reg.get_adapter("ice_only")


def test_is_atmosphere_mode():
    assert reg.is_atmosphere_mode("atmosphere")
    assert reg.is_atmosphere_mode("coupled_climate")
    assert not reg.is_atmosphere_mode("ocean")
    assert not reg.is_atmosphere_mode("ocean_only")


def test_load_adapter_atmosphere(tmp_path):
    p = tmp_path / "atm.yaml"
    p.write_text(textwrap.dedent(
        """
        model:
          type: atmosphere_only
        grid:
          type: cubed_sphere
          resolution: 48
          n_levels: 1
        atmosphere:
          dynamics: shallow_water
        """
    ))
    mode, cfg = reg.load_adapter(str(p))
    assert mode == "atmosphere_only"
    assert isinstance(cfg, Config)


def test_load_adapter_ocean(tmp_path):
    p = tmp_path / "ocn.yaml"
    p.write_text(textwrap.dedent(
        """
        model:
          type: ocean_only
        grid:
          type: latlon_cgrid
        ocean:
          A_h: 30000.0
        """
    ))
    mode, cfg = reg.load_adapter(str(p))
    assert mode == "ocean_only"
    assert isinstance(cfg, OceanExperimentConfig)
    assert cfg.get("ocean.A_h") == 30000.0


def test_uniform_protocol_satisfied_by_both_adapters():
    # Both adapters must satisfy the structural ConfigAdapter protocol.
    for inst in (Config(), OceanExperimentConfig()):
        assert isinstance(inst, reg.ConfigAdapter)
        # The methods init_experiment/validate_templates rely on.
        assert callable(inst.get_meta)
        assert callable(inst.signature)
        assert callable(inst.validate_strict)
        assert isinstance(inst.run_command(), str)
        assert isinstance(inst.signature(), str)


def test_known_modes_includes_both_components():
    modes = reg.known_modes()
    assert "atmosphere" in modes
    assert "ocean" in modes
