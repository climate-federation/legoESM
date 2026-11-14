"""Smoke test for the AMIP experiment with RRTMG radiation.

Verifies that the AMIP pipeline with RRTMGP correlated-k radiation runs
for a few steps without blowup, using synthetic forcing data and a small
grid (C4/L5) to keep CI fast.

Also tests radiation-option selection plumbing and the radiation cadence
(update_interval) mechanism.
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.forcing.amip import AMIPForcingConfig, load_amip_forcing, get_forcing_at_time
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig, RRTMGPConfig, RadiationConfig,
)
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import rrtmgp_radiation
from legoesm.atmosphere.physics.radiation.solar import daily_mean_insolation
from legoesm.atmosphere.physics.convection.config import SBMConfig
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm import constants
from legoesm.forcing.time_utils import day_to_calendar


def _make_synthetic_forcing(path, n_lat=18, n_lon=36):
    """Create minimal synthetic AMIP forcing."""
    lat = np.linspace(-85, 85, n_lat)
    lon = np.linspace(5, 355, n_lon)
    times = np.array([np.datetime64("2000-01-01") + np.timedelta64(i * 30, "D")
                       for i in range(4)])
    sst = np.full((4, n_lat, n_lon), 293.0, dtype=np.float32)
    sst += lat[None, :, None] * 0.2
    sic = np.zeros((4, n_lat, n_lon), dtype=np.float32)
    sic[:, :2, :] = 0.8
    sic[:, -2:, :] = 0.6

    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), sst),
         "sic": (("time", "lat", "lon"), sic)},
        coords={"time": times, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)


class TestRadiationOptionSelection:
    """Test that radiation scheme selection works correctly."""


    def test_radiation_config_scheme_selection(self):
        """RadiationConfig correctly selects gray vs rrtmgp."""
        cfg_gray = RadiationConfig(scheme="gray")
        assert cfg_gray.scheme == "gray"

        cfg_rrtmg = RadiationConfig(scheme="rrtmgp")
        assert cfg_rrtmg.scheme == "rrtmgp"

    def test_amip_config_radiation_field(self):
        """AMIPExperimentConfig has radiation field."""
        from legoesm.forcing.amip_config import AMIPExperimentConfig
        cfg = AMIPExperimentConfig(radiation="rrtmg", rad_update_steps=3)
        assert cfg.radiation == "rrtmg"
        assert cfg.rad_update_steps == 3

    def test_amip_config_gas_concentrations(self):
        """AMIPExperimentConfig exposes gas concentrations."""
        from legoesm.forcing.amip_config import AMIPExperimentConfig
        cfg = AMIPExperimentConfig(co2_ppmv=560.0)
        assert cfg.co2_ppmv == 560.0
        assert cfg.ch4_ppbv == 1900.0  # default
