"""Unit tests for :mod:`legoesm.training.run_to_column_mean`.

The run→time-mean→compare wiring: the generic accumulation logic is tested with
a mock driver (deterministic, cheap), and the real CMIP path with a tiny coupled
run (slow).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace  # noqa: E402

from legoesm.training.compare_reanalysis import ColumnState  # noqa: E402
from legoesm.training.run_to_column_mean import (  # noqa: E402
    amip_column_state,
    cmip_column_state,
    run_to_column_mean,
)


def _state(scale, *, nlat=2, nlon=3, nlev=4):
    shp = (nlat, nlon, nlev)
    sfc = (nlat, nlon)
    return ColumnState(
        T=jnp.full(shp, 280.0 + scale),
        q_v=jnp.full(shp, 5e-3),
        u=jnp.full(shp, scale),
        v=jnp.zeros(shp),
        p_s=jnp.full(sfc, 1.0e5),
        sst_K=jnp.full(sfc, 290.0 + scale),
    )


class _FakeDriver:
    """Mimics a driver: ``run(segment_callback=...)`` fires the callback once per
    segment, advancing an internal index so the extractor sees a new state."""

    def __init__(self, states):
        self.states = states
        self.i = -1
        self.run_kwargs = None

    def run(self, segment_callback, **kwargs):
        self.run_kwargs = kwargs
        for k in range(len(self.states)):
            self.i = k
            segment_callback(self, float(k), 1.0)
        return "OK"


def _extract(driver, day, dt):  # noqa: ARG001
    return driver.states[driver.i]


def test_run_to_column_mean_is_time_mean():
    states = [_state(0.0), _state(3.0), _state(6.0)]  # mean scale = 3
    driver = _FakeDriver(states)
    mean = run_to_column_mean(driver, _extract)
    np.testing.assert_allclose(np.asarray(mean.T), 283.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(mean.u), 3.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(mean.sst_K), 293.0, rtol=1e-12)


def test_run_to_column_mean_forwards_run_kwargs():
    driver = _FakeDriver([_state(1.0)])
    run_to_column_mean(driver, _extract, run_kwargs={"start_day": 7.0})
    assert driver.run_kwargs == {"start_day": 7.0}


def test_run_to_column_mean_no_segment_raises():
    """A run that fires no segment boundary must raise, not return a zero mean."""
    driver = _FakeDriver([])  # run() fires the callback zero times
    with pytest.raises(ValueError, match="no segment boundary fired"):
        run_to_column_mean(driver, _extract)


def test_run_to_column_mean_single_segment():
    driver = _FakeDriver([_state(4.0)])
    mean = run_to_column_mean(driver, _extract)
    np.testing.assert_allclose(np.asarray(mean.T), 284.0, rtol=1e-12)


def test_run_to_column_mean_threads_segment_day():
    """The segment `day` is threaded into the extractor (NOT day-0 every time) —
    AMIP SST is time-dependent, so a multi-day mean must see each segment's day."""
    seen_days = []

    def extract(driver, day, dt):
        seen_days.append(day)
        return driver.states[driver.i]

    driver = _FakeDriver([_state(0.0), _state(1.0), _state(2.0)])
    run_to_column_mean(driver, extract)
    assert seen_days == [0.0, 1.0, 2.0]


class _FakeAmipDriver:
    """Minimal AMIP driver: fixed atm state, TIME-DEPENDENT prescribed SST."""

    def __init__(self, n_seg, nlat=2, nlon=3, nlev=4):
        shp, sfc = (nlat, nlon, nlev), (nlat, nlon)
        self.state = SimpleNamespace(
            T=jnp.full(shp, 280.0), u=jnp.zeros(shp),
            v=jnp.zeros(shp), p_s=jnp.full(sfc, 1.0e5),
        )
        self.q_v = jnp.full(shp, 5e-3)
        self._n_seg = n_seg
        self._sfc = sfc

    def get_sst_sic(self, day):
        # SST rises with the calendar day → mean must reflect all sampled days.
        return jnp.full(self._sfc, 290.0 + day), None

    def run(self, segment_callback, **kwargs):  # noqa: ARG002
        for k in range(self._n_seg):
            segment_callback(self, float(k), 1.0)
        return "OK"


def test_amip_prescribed_sst_threaded_through_segments():
    """The AMIP day-threading bug guard: with day-varying prescribed SST, the
    time-mean SST is the mean over the SAMPLED days, not day-0 repeated."""
    driver = _FakeAmipDriver(n_seg=3)  # days 0,1,2 → SST 290,291,292
    mean = run_to_column_mean(driver, amip_column_state)
    np.testing.assert_allclose(np.asarray(mean.sst_K), 291.0, rtol=1e-12)  # mean(290,291,292)
    # (A day-0-only bug would give 290.0.)


def _tiny_coupled(days, diag_days, n_lat=8, nlev=5):
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
        OutputConfig,
    )
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from legoesm.grids.latlon import create_latlon_grid

    atm_config = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=n_lat, nlev=nlev),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        output=OutputConfig(diag_days=diag_days),
        radiation="gray", days=days,
    )
    ocean_grid = create_latlon_grid(n_lat=n_lat, n_lon=2 * n_lat)
    driver = CoupledESMDriver(atm_config, PRESETS["aquaplanet"](), ocean_grid=ocean_grid)
    driver.setup()
    return driver


@pytest.mark.slow
def test_run_to_column_mean_real_cmip():
    """A real tiny coupled (CMIP) run produces a finite time-mean ColumnState
    with the coupled SST; multiple diagnostic segments are sampled + averaged."""
    driver = _tiny_coupled(days=1.0, diag_days=0.5)  # ~2 segments
    n_samples = {"n": 0}

    def counting_extract(d, day, dt):
        n_samples["n"] += 1
        return cmip_column_state(d, day, dt)

    mean = run_to_column_mean(driver, counting_extract)
    assert n_samples["n"] > 1  # multiple diagnostic segments sampled + averaged
    assert isinstance(mean, ColumnState)
    n_lat, n_lon, nlev = mean.T.shape
    assert (n_lat, n_lon, nlev) == (8, 16, 5)
    for name in ("T", "q_v", "u", "v", "p_s"):
        assert bool(jnp.all(jnp.isfinite(getattr(mean, name))))
    # CMIP: the coupled-ocean SST is carried through.
    assert mean.sst_K is not None
    assert mean.sst_K.shape == (8, 16)
    assert bool(jnp.all(jnp.isfinite(mean.sst_K)))


@pytest.mark.slow
def test_segment_callback_compose_is_optional():
    """CoupledESMDriver.run with no segment_callback still runs (byte-identical
    coupling path); the new optional hook does not break the plain run."""
    driver = _tiny_coupled(days=0.5, diag_days=0.5)
    status = driver.run()  # no segment_callback → plain coupled run
    assert isinstance(status, str)
