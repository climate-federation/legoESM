"""Held-Suarez with surface_bulk_scheme="none": the single-device lat-lon driver
and the 2-device lat-band lane take the same first step.

The lat-band lane applies no surface fluxes; before "none" existed the
single-device pipeline added constant-coefficient bulk fluxes regardless, a
~9e-3 K bottom-level gap after one step.  Needs
``XLA_FLAGS=--xla_force_host_platform_device_count=2``.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
from legoesm.driver.model_driver import ModelDriver

_DT = 100.0


def _T_after_one_step(tmp_path, *, spmd, sbs):
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=4),
        dycore=DycoreConfig(model_type="hydrostatic",
                            discretization="latlon_cgrid", dt=_DT, fix_mass=True),
        output=OutputConfig(output_dir=str(tmp_path / f"{sbs}{spmd}"),
                            diag_days=0, checkpoint_days=0),
        days=_DT / 86400.0, dataset="analytical", held_suarez_forcing=True,
        turbulence="none", radiation="none", convection="none",
        microphysics="none", gravity_wave_drag="none", cloud_scheme="none",
        fix_moisture=True, enable_latlon_spmd=spmd, distributed=False,
        n_devices=2 if spmd else 1, surface_bulk_scheme=sbs)
    d = ModelDriver(cfg, output_dir=cfg.output.output_dir)
    d.setup()
    assert d.run(compiled=True) == "COMPLETED"
    return np.asarray(d.state.T.data, np.float64)


def test_one_vs_two_devices_agree_under_none(tmp_path):
    if len(jax.devices()) < 2:
        pytest.skip("needs --xla_force_host_platform_device_count=2")
    one = _T_after_one_step(tmp_path, spmd=False, sbs="none")
    two = _T_after_one_step(tmp_path, spmd=True, sbs="none")
    # One fp32 rounding step at ~300 K is 3.05e-5 K.
    assert np.abs(one - two).max() < 1e-4
