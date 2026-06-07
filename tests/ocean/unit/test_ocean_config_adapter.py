"""Unit tests for the ocean YAML config adapter (``OceanExperimentConfig``).

Mirrors the atmosphere ``legoesm.config.Config`` boundary tests: YAML load +
deep-merge, dot-notation get/set, runtime-config mapping, typo detection, and
strict validation that delegates to the runtime model validator.
"""
from __future__ import annotations

import textwrap

import pytest

from legoesm.ocean.config import OceanExperimentConfig, DEFAULT_OCEAN_CONFIG
from legoesm.ocean.state import (
    LatLonCGridOceanConfig,
    OceanConfig,
    SpectralOceanConfig,
)


def _write(tmp_path, text: str) -> str:
    p = tmp_path / "ocean.yaml"
    p.write_text(textwrap.dedent(text))
    return str(p)


def test_default_config_resolves_to_latlon_namedtuple():
    cfg = OceanExperimentConfig()
    rt = cfg.to_ocean_config()
    assert isinstance(rt, LatLonCGridOceanConfig)
    # Empty ocean: {} means every runtime default is preserved.
    assert rt == LatLonCGridOceanConfig()


def test_from_yaml_deep_merges_onto_defaults(tmp_path):
    path = _write(
        tmp_path,
        """
        grid:
          type: latlon_cgrid
          n_lat: 180
        ocean:
          A_h: 30000.0
          eos: wright
        time:
          dt_seconds: 1800
        """,
    )
    cfg = OceanExperimentConfig.from_yaml(path)
    # Overridden values present...
    assert cfg.get("grid.n_lat") == 180
    assert cfg.get("ocean.A_h") == 30000.0
    assert cfg.get("time.dt_seconds") == 1800
    # ...and defaults that were not overridden are retained.
    assert cfg.get("grid.n_lon") == DEFAULT_OCEAN_CONFIG["grid"]["n_lon"]
    rt = cfg.to_ocean_config()
    assert rt.A_h == 30000.0
    assert rt.eos == "wright"


def test_dot_get_set_roundtrip():
    cfg = OceanExperimentConfig()
    cfg.set("ocean.bottom_drag_r", 2.5e-3)
    assert cfg.get("ocean.bottom_drag_r") == 2.5e-3
    assert cfg.to_ocean_config().bottom_drag_r == 2.5e-3
    assert cfg.get("nonexistent.key", "fallback") == "fallback"


def test_unknown_ocean_field_raises():
    cfg = OceanExperimentConfig.from_dict({"ocean": {"A_h_typo": 1.0}})
    with pytest.raises(ValueError, match="unknown ocean config field"):
        cfg.to_ocean_config()


def test_unknown_grid_type_raises():
    cfg = OceanExperimentConfig.from_dict({"grid": {"type": "octahedral"}})
    with pytest.raises(ValueError, match="grid.type must be one of"):
        cfg.to_ocean_config()


def test_cubed_sphere_maps_to_oceanconfig():
    cfg = OceanExperimentConfig.from_dict(
        {"grid": {"type": "cubed_sphere"}, "ocean": {"A_h": 1.2e4}}
    )
    rt = cfg.to_ocean_config()
    assert isinstance(rt, OceanConfig)
    assert rt.A_h == 1.2e4


def test_spectral_maps_to_spectraloceanconfig():
    cfg = OceanExperimentConfig.from_dict(
        {"grid": {"type": "spectral"}, "ocean": {"hyperdiff_coeff": 1.0e15}}
    )
    rt = cfg.to_ocean_config()
    assert isinstance(rt, SpectralOceanConfig)


def test_eos_linear_subdict_builds_namedtuple():
    from legoesm.ocean.eos import LinearEOSConfig

    cfg = OceanExperimentConfig.from_dict(
        {"ocean": {"eos": "linear", "eos_linear": {"alpha_T": 2.0e-4}}}
    )
    rt = cfg.to_ocean_config()
    assert rt.eos == "linear"
    assert isinstance(rt.eos_linear, LinearEOSConfig)
    assert rt.eos_linear.alpha_T == 2.0e-4


def test_eos_linear_unknown_field_raises():
    cfg = OceanExperimentConfig.from_dict(
        {"ocean": {"eos": "linear", "eos_linear": {"alpha_typo": 1.0}}}
    )
    with pytest.raises(ValueError, match="unknown ocean.eos_linear field"):
        cfg.to_ocean_config()


def test_nested_physics_block_rejected():
    cfg = OceanExperimentConfig.from_dict(
        {"ocean": {"physics": {"vertical_mixing": {}}}}
    )
    with pytest.raises(ValueError, match="not yet configurable via YAML"):
        cfg.to_ocean_config()


def test_validate_strict_passes_for_default():
    OceanExperimentConfig().validate_strict()  # must not raise


def test_validate_strict_rejects_bad_eos():
    cfg = OceanExperimentConfig.from_dict({"ocean": {"eos": "nonsense"}})
    with pytest.raises(ValueError, match="eos must be one of"):
        cfg.validate_strict()


def test_validate_strict_rejects_bad_barotropic_solver():
    cfg = OceanExperimentConfig.from_dict(
        {"ocean": {"barotropic_solver": "typo_solver"}}
    )
    with pytest.raises(ValueError, match="barotropic_solver must be one of"):
        cfg.validate_strict()


def test_validate_strict_rejects_nonpositive_dt():
    cfg = OceanExperimentConfig.from_dict({"time": {"dt_seconds": 0}})
    with pytest.raises(ValueError, match="time.dt_seconds must be > 0"):
        cfg.validate_strict()


def test_signature_changes_with_runtime_override_and_detects_noop():
    cfg = OceanExperimentConfig()
    sig0 = cfg.signature()
    cfg.set("ocean.A_h", 9.9e4)
    assert cfg.signature() != sig0
    # A non-runtime dot-path is a no-op on the resolved config signature.
    cfg2 = OceanExperimentConfig()
    sig_a = cfg2.signature()
    cfg2.set("ocean.totally_unused_metadata_path", 1)
    # Unknown field makes to_ocean_config raise -> unresolvable signature differs
    # from the resolvable one; the no-op detector keys off byte-identity, so this
    # is correctly NOT byte-identical (it is flagged as a change/typo upstream).
    assert cfg2.signature() != sig_a


def test_time_override_not_a_noop():
    cfg = OceanExperimentConfig()
    sig0 = cfg.signature()
    cfg.set("time.dt_seconds", 900)
    assert cfg.signature() != sig0


def test_yaml_roundtrip(tmp_path):
    cfg = OceanExperimentConfig.from_dict({"ocean": {"A_h": 2.0e4}})
    out = tmp_path / "out.yaml"
    cfg.to_yaml(str(out))
    reloaded = OceanExperimentConfig.from_yaml(str(out))
    assert reloaded.get("ocean.A_h") == 2.0e4
    assert reloaded.to_ocean_config() == cfg.to_ocean_config()
