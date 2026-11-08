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


def _driver(tmp_path, days=1.5, diag_days=1, checkpoint_days=0):
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="gaussian", resolution=8, nlev=8),
        dycore=DycoreConfig(energy_fixer=False, model_type="hydrostatic",
                            discretization="spectral", dt=DT),
        output=OutputConfig(output_dir="", diag_days=diag_days,
                            checkpoint_days=checkpoint_days),
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
    assert [n for n, _ in exports] == [48, 72]      # day 1, then the half day
    prev = 0
    for n, seg in exports:
        expect = np.mean(rates[prev:n], axis=0)
        assert float(expect.max()) > 0.0      # it rained (non-vacuity)
        np.testing.assert_allclose(seg, expect, rtol=1e-12, atol=0.0)
        prev = n
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
    assert seen == [None, None]


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


def test_coupling_boundary_closes_the_whole_day_and_the_partial_tail(tmp_path):
    """#2290: the day-1 boundary fired before the 48th step (the coupler saw
    47 steps of state, was told 86400 s) and a trailing half day was never
    handed to the coupler.  Each export must follow the step that ends its
    segment and report that segment's real length and end day."""
    d = _driver(tmp_path)
    steps, calls = [], []
    real_step = d.model.step

    def step(*a, **k):
        out = real_step(*a, **k)
        steps.append(out)
        return out

    def callback(driver, day, seconds):
        calls.append((len(steps), day, seconds))
        assert driver.state is steps[-1]          # state after the last step

    d.model.step = step
    assert d.run(segment_callback=callback) == "COMPLETED"
    assert [c[0] for c in calls] == [48, 72]
    np.testing.assert_allclose([c[1] for c in calls], [1.0, 1.5], rtol=0, atol=1e-12)
    assert [c[2] for c in calls] == [86400.0, 43200.0]


def test_each_checkpoint_holds_the_post_coupling_state(tmp_path):
    """A checkpoint on a coupling step (day 1 and the run's last step) must be
    written after the coupler stepped, so a restart neither skips nor repeats
    that coupling."""
    d = _driver(tmp_path, checkpoint_days=0.5)
    events = []
    real_save = d.save_checkpoint
    d.save_checkpoint = lambda step, day: (events.append(("ckpt", step)),
                                           real_save(step, day))
    nsteps = []
    real_step = d.model.step

    def step(*a, **k):
        nsteps.append(1)
        return real_step(*a, **k)
    d.model.step = step
    assert d.run(segment_callback=lambda *a: events.append(("cpl", len(nsteps)))) \
        == "COMPLETED"
    assert events == [("ckpt", 24), ("cpl", 48), ("ckpt", 48),
                      ("cpl", 72), ("ckpt", 72)]


def test_a_nonfinite_state_never_reaches_the_coupler(tmp_path):
    """The diagnostic blow-up check runs every diag_days only; the coupling
    boundary must refuse a non-finite state on its own."""
    import jax
    d = _driver(tmp_path, diag_days=2)
    real_step = d.model.step
    n = []

    def step(*a, **k):
        out = real_step(*a, **k)
        n.append(1)
        return jax.tree.map(lambda x: x * np.nan, out) if len(n) == 48 else out
    d.model.step = step
    calls = []
    status = d.run(segment_callback=lambda *a: calls.append(1))
    assert status.startswith("BLOWUP") and calls == []


def test_a_nonfinite_tracer_never_reaches_the_coupler(tmp_path):
    import jax
    d = _driver(tmp_path, diag_days=2)
    real_step = d.model.step
    n = []

    def step(*a, **k):
        out = real_step(*a, **k)
        n.append(1)
        if len(n) == 48:
            assert "q_v" in out.tracers              # non-vacuity: moist run
            out = out._replace(tracers=jax.tree.map(
                lambda x: x * np.nan, out.tracers))
        return out
    d.model.step = step
    calls = []
    status = d.run(segment_callback=lambda *a: calls.append(1))
    assert status.startswith("BLOWUP") and calls == []
