"""Round-trip tests for the unified configuration system.

Validates:
- ExperimentConfig ↔ JSON serialization round-trip.
- ExperimentConfig ↔ AMIPExperimentConfig conversion preserves all fields.
- YAML Config → ExperimentConfig → JSON → ExperimentConfig round-trip.
- Legacy AMIPExperimentConfig checkpoint → load → ExperimentConfig path.
- ExperimentConfig JSON save/load.
- experiment_config_from_dict ignores unknown fields.
- create_experiment_config returns ExperimentConfig (not AMIP).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from legoesm.config import Config
from legoesm.driver.config import (
    ExperimentConfig,
    GridConfig,
    DycoreConfig,
    OutputConfig,
    experiment_config_to_dict,
    experiment_config_from_dict,
    save_experiment_config,
    load_experiment_config,
)
from legoesm.forcing.amip_config import (
    AMIPExperimentConfig,
    config_to_dict as amip_config_to_dict,
    config_from_dict as amip_config_from_dict,
)


# ===========================================================================
# 1. ExperimentConfig ↔ JSON round-trip
# ===========================================================================

class TestExperimentConfigJSON:
    """Native JSON serialization of ExperimentConfig."""

    def test_default_roundtrip(self):
        cfg = ExperimentConfig()
        d = experiment_config_to_dict(cfg)
        restored = experiment_config_from_dict(d)
        assert cfg == restored

    def test_custom_fields_roundtrip(self):
        cfg = ExperimentConfig(
            grid=GridConfig(resolution=48, nlev=32, grid_type="cubed_sphere"),
            dycore=DycoreConfig(dt=300.0, model_type="hydrostatic"),
            output=OutputConfig(diag_days=10, checkpoint_days=30),
            days=365,
            co2_ppmv=560.0,
            radiation="rrtmg",
            microphysics="kessler",
            topography="/path/to/topo.nc",
            distributed=True,
            ensemble_size=4,
        )
        d = experiment_config_to_dict(cfg)
        restored = experiment_config_from_dict(d)
        assert cfg == restored

    def test_dict_is_json_safe(self):
        cfg = ExperimentConfig(
            grid=GridConfig(resolution=24),
            days=100,
        )
        d = experiment_config_to_dict(cfg)
        # Should be JSON-serializable
        text = json.dumps(d)
        restored_d = json.loads(text)
        restored = experiment_config_from_dict(restored_d)
        assert restored.grid.resolution == 24
        assert restored.days == 100

    def test_sub_configs_are_nested_dicts(self):
        cfg = ExperimentConfig()
        d = experiment_config_to_dict(cfg)
        assert isinstance(d["grid"], dict)
        assert isinstance(d["dycore"], dict)
        assert isinstance(d["output"], dict)
        assert "resolution" in d["grid"]
        assert "dt" in d["dycore"]
        assert "diag_days" in d["output"]

    def test_unknown_fields_dropped(self):
        """Extra fields in input dict should be silently ignored."""
        d = experiment_config_to_dict(ExperimentConfig())
        d["some_future_field"] = True
        d["grid"]["some_grid_field"] = 42
        restored = experiment_config_from_dict(d)
        assert isinstance(restored, ExperimentConfig)
        assert not hasattr(restored, "some_future_field")

    def test_file_save_load_roundtrip(self):
        cfg = ExperimentConfig(
            grid=GridConfig(resolution=96, nlev=60),
            dycore=DycoreConfig(dt=150.0),
            co2_ppmv=800.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.json"
            save_experiment_config(cfg, path)
            restored = load_experiment_config(path)
        assert cfg == restored

    def test_all_fields_survive_json(self):
        """Every field of ExperimentConfig survives JSON serialization."""
        cfg = ExperimentConfig()
        d = experiment_config_to_dict(cfg)
        text = json.dumps(d, default=str)
        restored = experiment_config_from_dict(json.loads(text))

        for field in ExperimentConfig._fields:
            orig = getattr(cfg, field)
            rest = getattr(restored, field)
            assert orig == rest, f"Field {field}: {orig!r} != {rest!r}"


# ===========================================================================
# 2. ExperimentConfig ↔ AMIPExperimentConfig conversion
# ===========================================================================

class TestAMIPConversion:
    """Bi-directional conversion between ExperimentConfig and AMIPExperimentConfig."""

    def test_to_amip_roundtrip(self):
        """ExperimentConfig → AMIP → ExperimentConfig preserves fields."""
        cfg = ExperimentConfig(
            grid=GridConfig(resolution=48, nlev=40),
            dycore=DycoreConfig(dt=300.0),
            output=OutputConfig(diag_days=5, checkpoint_days=30,
                                monthly_means=True, cmip_output=True),
            days=365,
            co2_ppmv=560.0,
            radiation="rrtmg",
            distributed=True,
        )
        amip = cfg.to_amip_config()
        restored = ExperimentConfig.from_amip_config(amip)

        # Check key fields survive roundtrip
        assert restored.grid.resolution == 48
        assert restored.grid.nlev == 40
        assert restored.dycore.dt == 300.0
        assert restored.output.diag_days == 5
        assert restored.output.checkpoint_days == 30
        assert restored.output.monthly_means is True
        assert restored.output.cmip_output is True
        assert restored.days == 365
        assert restored.co2_ppmv == 560.0
        assert restored.radiation == "rrtmg"
        assert restored.distributed is True

    def test_from_amip_roundtrip(self):
        """AMIPExperimentConfig → ExperimentConfig → AMIP preserves fields."""
        amip = AMIPExperimentConfig(
            resolution=24,
            nlev=20,
            dt=1200.0,
            days=100,
            radiation="gray",
            co2_ppmv=284.3,
            microphysics="kessler",
            S_0=1360.0,
            distributed=False,
        )
        cfg = ExperimentConfig.from_amip_config(amip)
        restored_amip = cfg.to_amip_config()

        assert restored_amip.resolution == 24
        assert restored_amip.nlev == 20
        assert restored_amip.dt == 1200.0
        assert restored_amip.days == 100
        assert restored_amip.co2_ppmv == 284.3
        assert restored_amip.microphysics == "kessler"
        assert restored_amip.S_0 == 1360.0

    def test_all_shared_fields_survive(self):
        """All fields shared between ExperimentConfig and AMIPExperimentConfig
        survive the round-trip."""
        cfg = ExperimentConfig()
        amip = cfg.to_amip_config()
        restored = ExperimentConfig.from_amip_config(amip)

        # These are the scalar fields on ExperimentConfig that have direct
        # counterparts on AMIPExperimentConfig
        for field in ExperimentConfig._fields:
            if field in ("grid", "dycore", "output"):
                continue  # sub-configs tested separately
            orig = getattr(cfg, field)
            rest = getattr(restored, field)
            assert orig == rest, f"Field {field}: {orig!r} != {rest!r}"

    def test_amip_json_roundtrip_via_experiment(self):
        """AMIP JSON → dict → AMIPConfig → ExperimentConfig → AMIP → dict
        preserves all fields."""
        amip = AMIPExperimentConfig(resolution=32, days=200, co2_ppmv=450.0)
        d = amip_config_to_dict(amip)
        text = json.dumps(d)
        d2 = json.loads(text)
        amip2 = amip_config_from_dict(d2)
        cfg = ExperimentConfig.from_amip_config(amip2)
        amip3 = cfg.to_amip_config()

        assert amip3.resolution == 32
        assert amip3.days == 200
        assert amip3.co2_ppmv == 450.0


# ===========================================================================
# 3. YAML Config → ExperimentConfig
# ===========================================================================

class TestYAMLToExperimentConfig:
    """YAML → Config → ExperimentConfig."""

    def test_default_yaml_config(self):
        cfg = Config()
        exp = cfg.to_experiment_config()
        assert isinstance(exp, ExperimentConfig)
        assert exp.grid.grid_type == "cubed_sphere"

    def test_yaml_roundtrip(self):
        """YAML → Config → ExperimentConfig → JSON → ExperimentConfig."""
        cfg = Config.from_dict({
            "grid": {"resolution": 32, "n_levels": 20},
            "atmosphere": {"dt_seconds": 300, "dynamics": "hydrostatic"},
            "time": {"duration_hours": 240},
        })
        exp = cfg.to_experiment_config()

        d = experiment_config_to_dict(exp)
        restored = experiment_config_from_dict(d)

        assert restored.grid.resolution == 32
        assert restored.grid.nlev == 20
        assert restored.dycore.dt == 300.0
        assert restored.days == 10  # 240 / 24

    def test_yaml_file_roundtrip(self):
        """Save YAML → load → to_experiment_config → save JSON → load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            json_path = Path(tmpdir) / "test_config.json"

            cfg = Config.from_dict({
                "grid": {"resolution": 16, "n_levels": 5},
                "time": {"duration_hours": 48},
            })
            cfg.to_yaml(str(yaml_path))

            loaded = Config.from_yaml(str(yaml_path))
            exp = loaded.to_experiment_config()

            save_experiment_config(exp, json_path)
            restored = load_experiment_config(json_path)

            assert restored.grid.resolution == 16
            assert restored.grid.nlev == 5
            assert restored.days == 2  # 48 / 24


# ===========================================================================
# 4. Experiment factory returns ExperimentConfig
# ===========================================================================

class TestExperimentFactory:
    """create_experiment_config returns canonical ExperimentConfig."""

    def test_returns_experiment_config(self):
        from legoesm.forcing.experiments import create_experiment_config
        cfg = create_experiment_config("piControl", resolution=16)
        assert isinstance(cfg, ExperimentConfig)

    def test_amip_factory_returns_amip_config(self):
        import warnings
        from legoesm.forcing.experiments import create_amip_experiment_config
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = create_amip_experiment_config("piControl", resolution=16)
        assert isinstance(cfg, AMIPExperimentConfig)

    def test_factory_ghg_values(self):
        from legoesm.forcing.experiments import create_experiment_config
        cfg = create_experiment_config("piControl")
        assert cfg.co2_ppmv == pytest.approx(284.3, abs=0.1)

    def test_factory_override_applied(self):
        from legoesm.forcing.experiments import create_experiment_config
        cfg = create_experiment_config("piControl", resolution=96)
        assert cfg.grid.resolution == 96

    def test_factory_roundtrip_json(self):
        from legoesm.forcing.experiments import create_experiment_config
        cfg = create_experiment_config("historical", resolution=48)

        d = experiment_config_to_dict(cfg)
        restored = experiment_config_from_dict(d)

        assert restored.co2_ppmv == cfg.co2_ppmv
        assert restored.grid.resolution == 48


# ===========================================================================
# 5. Backward compatibility: restart IO accepts ExperimentConfig
# ===========================================================================

class TestRestartConfigCompat:
    """restart.py config_to_dict / compute_config_hash accept both types."""

    def test_config_to_dict_experiment_config(self):
        from legoesm.io.restart import config_to_dict
        cfg = ExperimentConfig(days=100, co2_ppmv=560.0)
        d = config_to_dict(cfg)
        assert d["days"] == 100
        assert d["co2_ppmv"] == 560.0

    def test_config_to_dict_amip_config(self):
        from legoesm.io.restart import config_to_dict
        cfg = AMIPExperimentConfig(days=100, co2_ppmv=560.0)
        d = config_to_dict(cfg)
        assert d["days"] == 100
        assert d["co2_ppmv"] == 560.0

    def test_config_hash_experiment_config(self):
        from legoesm.io.restart import compute_config_hash
        cfg = ExperimentConfig(days=100)
        h = compute_config_hash(cfg)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_config_hash_amip_config(self):
        from legoesm.io.restart import compute_config_hash
        cfg = AMIPExperimentConfig(days=100)
        h = compute_config_hash(cfg)
        assert isinstance(h, str)
        assert len(h) == 64


# ===========================================================================
# 6. Default value consistency
# ===========================================================================

class TestDefaultConsistency:
    """Fields shared between ExperimentConfig and AMIPExperimentConfig
    should have matching defaults where names are the same."""

    def test_shared_scalar_defaults(self):
        """Non-sub-config fields with identical names have matching defaults."""
        exp = ExperimentConfig()
        amip = AMIPExperimentConfig()

        # Fields with the same name on both
        shared_fields = set(ExperimentConfig._fields) & set(AMIPExperimentConfig._fields)
        shared_fields -= {"grid", "dycore", "output"}  # sub-configs don't exist on AMIP

        # Some fields have intentionally different defaults between the two
        # (legacy AMIP has different surface parameter defaults)
        intentionally_different = {
            "dataset",     # "analytical" vs "cobe"
            "T_init",      # 300.0 vs 280.0
            "RH_init",     # 0.7 vs 0.6
            "S_0",         # 1361.0 vs 1360.0
            "C_H",         # 0.0044 vs 1.5e-3
            "C_E",         # 0.0044 vs 1.5e-3
            "sfc_emissivity",  # 0.97 vs 0.98
            "emissivity_ice",  # 0.95 vs 0.99
        }

        for field in sorted(shared_fields - intentionally_different):
            exp_val = getattr(exp, field)
            amip_val = getattr(amip, field)
            assert exp_val == amip_val, (
                f"Default mismatch for {field}: "
                f"ExperimentConfig={exp_val!r} vs AMIPExperimentConfig={amip_val!r}"
            )
