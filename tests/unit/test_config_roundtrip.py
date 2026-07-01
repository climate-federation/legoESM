"""Round-trip tests for the unified configuration system."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from legoesm.driver.config import (
    DycoreConfig,
    EvaluationConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
    experiment_config_from_dict,
    experiment_config_to_dict,
    load_experiment_config,
    save_experiment_config,
)
from legoesm.forcing.amip_config import (
    AMIPExperimentConfig,
)
from legoesm.forcing.amip_config import (
    config_from_dict as amip_config_from_dict,
)
from legoesm.forcing.amip_config import (
    config_to_dict as amip_config_to_dict,
)


class TestExperimentConfigJSON:
    def test_default_roundtrip(self):
        cfg = ExperimentConfig()
        restored = experiment_config_from_dict(experiment_config_to_dict(cfg))
        assert cfg == restored

    def test_joint_fields_roundtrip(self):
        cfg = ExperimentConfig(
            grid=GridConfig(resolution=48, nlev=32, grid_type="cubed_sphere"),
            dycore=DycoreConfig(dt=300.0, model_type="hydrostatic"),
            output=OutputConfig(diag_days=10, checkpoint_days=30),
            convection="mass_flux",
            turbulence="louis",
            physics_parameterization="ml",
            physics_parameterization_checkpoint="checkpoints/physics.eqx",
            physics_parameterization_stats="checkpoints/physics_stats.npz",
            physics_parameterization_hidden_dim=192,
            physics_parameterization_layers=4,
            physics_parameterization_seed=7,
        )
        restored = experiment_config_from_dict(experiment_config_to_dict(cfg))
        assert cfg == restored

    def test_nested_evaluation_config_roundtrip(self):
        """OutputConfig.evaluation is a NamedTuple nested one level deeper
        than grid/dycore/output — without explicit handling it survives
        json.dumps as a bare positional list (NamedTuple is a tuple),
        losing field names on reload. ``suites`` is additionally a tuple
        field that JSON round-trips as a list, so it must be coerced back
        to a tuple or the reconstructed config != the original."""
        cfg = ExperimentConfig(
            output=OutputConfig(
                cmip_output=True,
                evaluation=EvaluationConfig(
                    enabled=True,
                    suites=("Tier1_sanity_checks", "Tier2_atmosphere_monthly"),
                    climateeval_python="/opt/climateeval/bin/python",
                    data_root_dir="/data/climateeval",
                    timerange="19790101/19791231",
                ),
            )
        )
        restored = experiment_config_from_dict(experiment_config_to_dict(cfg))
        assert cfg == restored
        assert isinstance(restored.output.evaluation, EvaluationConfig)
        assert restored.output.evaluation.suites == (
            "Tier1_sanity_checks", "Tier2_atmosphere_monthly")

    def test_file_save_load_roundtrip(self):
        cfg = ExperimentConfig(
            grid=GridConfig(resolution=96, nlev=60),
            dycore=DycoreConfig(dt=150.0),
            physics_parameterization="ml",
            physics_parameterization_checkpoint="physics.eqx",
            physics_parameterization_stats="physics_stats.npz",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.json"
            save_experiment_config(cfg, path)
            restored = load_experiment_config(path)
        assert cfg == restored


class TestCheckpointConfigAutodetect:
    """forcing.amip_config.config_from_dict must round-trip an ExperimentConfig.

    Regression: an NPZ checkpoint saved from an ExperimentConfig used to load back
    through the flat AMIP deserializer, silently dropping every non-AMIP field
    (including the Stage-A1 master RNG ``seed``).
    """

    def test_experiment_config_seed_survives_checkpoint_serializer(self):
        from legoesm.forcing.amip_config import (
            config_from_dict as ckpt_from_dict,
        )
        from legoesm.forcing.amip_config import (
            config_to_dict as ckpt_to_dict,
        )

        cfg = ExperimentConfig(
            grid=GridConfig(resolution=48, nlev=40),
            dycore=DycoreConfig(dt=300.0),
            seed=12345,
        )
        restored = ckpt_from_dict(ckpt_to_dict(cfg))
        assert isinstance(restored, ExperimentConfig)
        assert restored.seed == 12345
        assert restored == cfg

    def test_flat_amip_dict_still_loads_as_amip(self):
        from legoesm.forcing.amip_config import config_from_dict as ckpt_from_dict

        restored = ckpt_from_dict({"resolution": 24, "nlev": 20, "days": 100})
        assert isinstance(restored, AMIPExperimentConfig)
        assert restored.resolution == 24


class TestAMIPConversion:
    def test_to_amip_roundtrip(self):
        cfg = ExperimentConfig(
            grid=GridConfig(resolution=48, nlev=40),
            dycore=DycoreConfig(dt=300.0),
            output=OutputConfig(diag_days=5, checkpoint_days=30),
            convection="mass_flux",
            turbulence="louis",
            physics_parameterization="ml",
            physics_parameterization_checkpoint="physics.eqx",
            physics_parameterization_stats="physics_stats.npz",
            physics_parameterization_hidden_dim=192,
            physics_parameterization_layers=4,
            physics_parameterization_seed=7,
        )
        amip = cfg.to_amip_config()
        restored = ExperimentConfig.from_amip_config(amip)
        assert restored.convection == "mass_flux"
        assert restored.turbulence == "louis"
        assert restored.physics_parameterization == "ml"
        assert restored.physics_parameterization_checkpoint == "physics.eqx"
        assert restored.physics_parameterization_stats == "physics_stats.npz"
        assert restored.physics_parameterization_hidden_dim == 192
        assert restored.physics_parameterization_layers == 4
        assert restored.physics_parameterization_seed == 7

    def test_from_amip_roundtrip(self):
        amip = AMIPExperimentConfig(
            resolution=24,
            nlev=20,
            dt=1200.0,
            days=100,
            convection="mass_flux",
            turbulence="louis",
            physics_parameterization="ml",
            physics_parameterization_checkpoint="physics.eqx",
            physics_parameterization_stats="physics_stats.npz",
            S_0=1360.0,
            distributed=False,
        )
        cfg = ExperimentConfig.from_amip_config(amip)
        restored_amip = cfg.to_amip_config()
        assert restored_amip.physics_parameterization == "ml"
        assert restored_amip.physics_parameterization_checkpoint == "physics.eqx"
        assert restored_amip.physics_parameterization_stats == "physics_stats.npz"
        assert restored_amip.convection == "mass_flux"
        assert restored_amip.turbulence == "louis"

    def test_amip_json_roundtrip_via_experiment(self):
        amip = AMIPExperimentConfig(
            resolution=32,
            days=200,
            physics_parameterization="ml",
            physics_parameterization_checkpoint="physics.eqx",
            physics_parameterization_stats="physics_stats.npz",
        )
        d = amip_config_to_dict(amip)
        text = json.dumps(d)
        amip2 = amip_config_from_dict(json.loads(text))
        cfg = ExperimentConfig.from_amip_config(amip2)
        amip3 = cfg.to_amip_config()
        assert amip3.physics_parameterization == "ml"
        assert amip3.physics_parameterization_checkpoint == "physics.eqx"
        assert amip3.physics_parameterization_stats == "physics_stats.npz"
