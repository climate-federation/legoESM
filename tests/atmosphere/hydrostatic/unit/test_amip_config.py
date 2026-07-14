"""Tests for AMIP experiment config, checkpoint/restart, and external forcing.

Verifies:
- AMIPExperimentConfig serialization roundtrip (JSON)
- Checkpoint save/load reproduces state to floating-point tolerance
- ExternalForcingConfig defaults and GHG constant mode
"""

import json
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants

jax.config.update("jax_enable_x64", True)

from legoesm.forcing.amip_config import (
    AMIPExperimentConfig,
    config_to_dict,
    config_from_dict,
    save_config,
    load_config,
    save_checkpoint,
    load_checkpoint,
)
from legoesm.forcing.external import (
    ExternalForcingConfig,
    GHGConfig,
    OzoneConfig,
    AerosolConfig,
    SolarConfig,
    get_ghg_at_time,
    get_ozone_at_time,
    get_aerosol_at_time,
    get_tsi_at_time,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init


class TestAMIPExperimentConfig:
    def test_defaults(self):
        cfg = AMIPExperimentConfig()
        assert cfg.resolution == 16
        assert cfg.nlev == 40
        assert cfg.dt == 600.0
        assert cfg.checkpoint_days == 0

    def test_to_dict_roundtrip(self):
        cfg = AMIPExperimentConfig(resolution=24, days=365, S_0=constants.S_0)
        d = config_to_dict(cfg)
        cfg2 = config_from_dict(d)
        assert cfg == cfg2

    def test_json_roundtrip(self, tmp_path):
        cfg = AMIPExperimentConfig(
            resolution=32, nlev=30, dt=450.0, days=90,
            dataset="hadisst", forcing_path="/tmp/test.nc",
        )
        path = tmp_path / "config.json"
        save_config(cfg, path)
        cfg2 = load_config(path)
        assert cfg == cfg2

    def test_unknown_fields_ignored(self):
        d = config_to_dict(AMIPExperimentConfig())
        d["unknown_future_field"] = 42
        cfg = config_from_dict(d)
        assert cfg.resolution == 16  # default preserved

    def test_json_is_readable(self, tmp_path):
        cfg = AMIPExperimentConfig(resolution=48)
        path = tmp_path / "config.json"
        save_config(cfg, path)
        with open(path) as f:
            data = json.load(f)
        assert data["resolution"] == 48
        assert isinstance(data["dt"], float)


class TestCheckpointRestart:
    @pytest.fixture
    def setup(self):
        N, NLEV = 4, 5
        grid = create_cubed_sphere(N)
        sigma = create_sigma_coordinate(NLEV)
        state = held_suarez_init(grid, sigma, T_init=280.0)
        q_v = jnp.ones((6, N, N, NLEV)) * 0.005
        config = AMIPExperimentConfig(resolution=N, nlev=NLEV)
        return grid, sigma, state, q_v, config

    def test_save_load_roundtrip(self, setup, tmp_path):
        grid, sigma, state, q_v, config = setup
        ckpt_path = tmp_path / "checkpoint.npz"

        save_checkpoint(ckpt_path, state, q_v, step=100, day=10.0, config=config)
        state2, q_v2, step2, day2, config2, diag_acc, _, _, _ = load_checkpoint(
            ckpt_path, grid, sigma,
        )

        assert step2 == 100
        assert day2 == 10.0
        assert config2.resolution == config.resolution
        assert config2.nlev == config.nlev

        # State arrays match to floating-point tolerance
        np.testing.assert_allclose(state2.T.data, state.T.data, atol=1e-12)
        np.testing.assert_allclose(state2.u.data, state.u.data, atol=1e-12)
        np.testing.assert_allclose(state2.v.data, state.v.data, atol=1e-12)
        np.testing.assert_allclose(state2.p_s.data, state.p_s.data, atol=1e-12)
        np.testing.assert_allclose(q_v2, q_v, atol=1e-12)

    def test_diag_accumulators_preserved(self, setup, tmp_path):
        grid, sigma, state, q_v, config = setup
        ckpt_path = tmp_path / "checkpoint_diag.npz"

        accum = {
            "precip_total": np.ones((6, 4, 4)) * 3.14,
            "rad_total": np.zeros((6, 4, 4)),
        }
        save_checkpoint(ckpt_path, state, q_v, step=50, day=5.0,
                        config=config, diag_accumulators=accum)

        _, _, _, _, _, diag_acc, _, _, _ = load_checkpoint(ckpt_path, grid, sigma)
        assert "precip_total" in diag_acc
        assert "rad_total" in diag_acc
        np.testing.assert_allclose(diag_acc["precip_total"], 3.14, atol=1e-12)

    def test_empty_diag_accumulators(self, setup, tmp_path):
        grid, sigma, state, q_v, config = setup
        ckpt_path = tmp_path / "checkpoint_nodiag.npz"

        save_checkpoint(ckpt_path, state, q_v, step=1, day=0.1, config=config)
        _, _, _, _, _, diag_acc, _, _, _ = load_checkpoint(ckpt_path, grid, sigma)
        assert diag_acc == {}


class TestExternalForcingConfig:
    def test_defaults_inactive(self):
        ext = ExternalForcingConfig()
        assert not ext.ozone.enabled
        assert not ext.aerosol.enabled
        assert ext.ghg.source == "constant"
        assert ext.solar.source == "constant"

    def test_ghg_constant_mode(self):
        ghg = GHGConfig(co2_ppmv=400.0)
        result = get_ghg_at_time(ghg, day=100.0)
        assert result["co2_ppmv"] == 400.0
        assert result["ch4_ppbv"] == 1650.0  # default

    def test_ghg_file_mode_missing_path_raises(self):
        ghg = GHGConfig(source="file", path="")
        with pytest.raises(ValueError, match="path must be set"):
            get_ghg_at_time(ghg, day=0.0)

    def test_ghg_unknown_source_raises(self):
        ghg = GHGConfig(source="magic")
        with pytest.raises(ValueError, match="Unknown GHG source"):
            get_ghg_at_time(ghg, day=0.0)

    def test_ozone_disabled_returns_none(self):
        ozone = OzoneConfig(enabled=False)
        assert get_ozone_at_time(ozone, day=0.0) is None

    def test_ozone_enabled_missing_path_raises(self):
        ozone = OzoneConfig(enabled=True, path="")
        with pytest.raises(ValueError, match="path must be set"):
            get_ozone_at_time(ozone, day=0.0)

    def test_aerosol_disabled_returns_none(self):
        aerosol = AerosolConfig(enabled=False)
        assert get_aerosol_at_time(aerosol, day=0.0) is None

    def test_aerosol_enabled_missing_path_raises(self):
        aerosol = AerosolConfig(enabled=True, path="")
        with pytest.raises(ValueError, match="path must be set"):
            get_aerosol_at_time(aerosol, day=0.0)

    def test_tsi_constant(self):
        solar = SolarConfig(S_0=1365.0)
        assert get_tsi_at_time(solar, day=42.0) == 1365.0

    def test_tsi_file_mode_missing_path_raises(self):
        solar = SolarConfig(source="file", path="")
        with pytest.raises(ValueError, match="path must be set"):
            get_tsi_at_time(solar, day=0.0)
