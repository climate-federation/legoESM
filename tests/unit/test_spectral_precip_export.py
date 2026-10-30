"""Spectral driver loop exports the surface precip (#2277).

At each coupling boundary ``seg_precip`` is the mean of the per-step rates
the model published since the previous boundary (the compiled lanes'
segment-mean semantics), and the lightweight timeseries carries its
area-weighted global mean [mm/day]."""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
from legoesm.driver.model_driver import ModelDriver

DT = 1800.0


def _driver(tmp_path, days=1.5, diag_days=1):
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="gaussian", resolution=8, nlev=8),
        dycore=DycoreConfig(energy_fixer=False, model_type="hydrostatic",
                            discretization="spectral", dt=DT),
        output=OutputConfig(output_dir="", diag_days=diag_days,
                            checkpoint_days=0),
        days=days, dataset="analytical", radiation="gray", rh_init=1.3,
        microphysics="sundqvist", convection="none", turbulence="none",
        precision="fp64", distributed=False,
    )
    d = ModelDriver(cfg, output_dir=str(tmp_path))
    d.setup()
    return d


def test_seg_precip_is_the_mean_of_the_step_rates(tmp_path):
    d = _driver(tmp_path)
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


def _patched(d, sfc_fn):
    """Replace each step's published precip by ``sfc_fn(published)``."""
    real_step = d.model.step

    def step(*a, **k):
        out = real_step(*a, **k)
        d.model._sfc_precip = sfc_fn(d.model._sfc_precip)
        return out
    d.model.step = step


def test_a_step_without_precip_is_refused_at_the_boundary(tmp_path):
    d = _driver(tmp_path)
    _patched(d, lambda sfc: None)
    with pytest.raises(RuntimeError, match="published no surface"):
        d.run(segment_callback=lambda *a: None)


def test_unknown_frozen_part_is_exported_as_none(tmp_path):
    d = _driver(tmp_path)
    _patched(d, lambda sfc: (sfc[0], None))
    seen = []
    d.run(segment_callback=lambda drv, *a: seen.append(
        drv._carry_aux["seg_precip_frozen"]))
    assert seen == [None]


def test_short_run_sample_carries_the_run_mean_precip(tmp_path):
    d = _driver(tmp_path, days=0.25, diag_days=1)
    rates = []
    real_step = d.model.step

    def step(*a, **k):
        out = real_step(*a, **k)
        rates.append(np.asarray(d.model._sfc_precip[0]))
        return out
    d.model.step = step
    assert d.run() == "COMPLETED"
    from legoesm.diagnostics.energy_budget import area_weighted_mean
    expect = float(area_weighted_mean(np.mean(rates, axis=0),
                                      d.grid.grid_area)) * 86400.0
    ts = np.load(tmp_path / "timeseries.npz")
    assert expect > 0.0
    np.testing.assert_allclose(ts["precip"], [expect], rtol=1e-12)
