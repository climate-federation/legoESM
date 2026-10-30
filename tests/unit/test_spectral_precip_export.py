"""Spectral driver loop exports the surface precip (#2277).

At each coupling boundary ``seg_precip`` is the mean of the per-step rates
the model published since the previous boundary (the compiled lanes'
segment-mean semantics), and the lightweight timeseries carries its
area-weighted global mean [mm/day]."""
from __future__ import annotations

import numpy as np

from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
from legoesm.driver.model_driver import ModelDriver

DT = 1800.0


def test_seg_precip_is_the_mean_of_the_step_rates(tmp_path):
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="gaussian", resolution=8, nlev=8),
        dycore=DycoreConfig(energy_fixer=False, model_type="hydrostatic",
                            discretization="spectral", dt=DT),
        output=OutputConfig(output_dir="", diag_days=1, checkpoint_days=0),
        days=1.5, dataset="analytical", radiation="gray", rh_init=1.3,
        microphysics="sundqvist", convection="none", turbulence="none",
        precision="fp64", distributed=False,
    )
    d = ModelDriver(cfg, output_dir=str(tmp_path))
    d.setup()
    rates, exports = [], []
    real_step = d.model.step

    def step(*a, **k):
        out = real_step(*a, **k)
        rates.append(np.asarray(d.model._sfc_precip[0]))
        return out

    def callback(driver, day, seconds):
        exports.append((len(rates), np.asarray(driver._carry_aux["seg_precip"])))

    d.model.step = step
    assert d.run(segment_callback=callback) == "COMPLETED"
    assert len(exports) == 1
    n, seg = exports[0]
    expect = np.mean(rates[:n], axis=0)
    assert float(expect.max()) > 0.0          # it rained (non-vacuity)
    np.testing.assert_allclose(seg, expect, rtol=1e-12, atol=0.0)
    ts = np.load(tmp_path / "timeseries.npz")
    assert np.all(np.isfinite(ts["precip"])) and float(ts["precip"][0]) > 0.0
