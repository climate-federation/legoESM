"""An SFNO (Gaussian-grid) run refuses to restart from a lat-lon checkpoint,
even when the two grids have the same shape (10x20 lat-lon vs T5)."""
from __future__ import annotations

import glob

import jax
import pytest

from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
from legoesm.driver.model_driver import ModelDriver


def _cfg(out, days, grid, discretization):
    return ExperimentConfig(
        grid=grid,
        dycore=DycoreConfig(model_type="hydrostatic",
                            discretization=discretization, dt=600.0,
                            fix_mass=True,
                            energy_fixer=discretization != "sfno"),
        output=OutputConfig(output_dir=str(out), diag_days=0,
                            checkpoint_days=1),
        days=days, dataset="analytical", held_suarez_forcing=True,
        turbulence="none", radiation="none", convection="none",
        microphysics="none", gravity_wave_drag="none", cloud_scheme="none",
        fix_moisture=True, distributed=False, n_devices=1,
        surface_bulk_scheme="none")


_needs_x64 = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="Spectral/Gaussian grids require JAX_ENABLE_X64=True",
)


@_needs_x64
def test_sfno_refuses_a_latlon_checkpoint(tmp_path):
    a = ModelDriver(_cfg(tmp_path / "ll", 1,
                         GridConfig(grid_type="latlon", resolution=10, nlev=4),
                         "latlon_cgrid"))
    a.setup()
    assert a.run(compiled=True) == "COMPLETED"
    ckpt = sorted(glob.glob(str(tmp_path / "ll" / "checkpoint_day_*.npz")))[-1]
    d = ModelDriver(_cfg(tmp_path / "sfno", 2,
                         GridConfig(grid_type="gaussian", resolution=5, nlev=4),
                         "sfno"))
    d.setup()
    with pytest.raises(ValueError, match="written by a lat-lon run"):
        d.load_checkpoint(ckpt)
