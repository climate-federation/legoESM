"""The MPAS lane must record its water-budget closure, and must not fake it.

This exists because the check was absent rather than wrong: the tracker lives
inside a collector method the MPAS run loop never calls, so every run of the
AMIP campaign published a blank moisture residual while a ~0.4 mm/day gap
between reported global evaporation and rainfall went unexamined. A test that
only asserted "the residual is small" would have passed against that blank.
So the assertions here are about the WIRING: that a known imbalance comes out
of the tracker with the right sign and size, and that the multi-rank path
refuses to publish a rank-local number instead of quietly publishing one.
"""
from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from legoesm import constants  # noqa: E402
from legoesm.diagnostics.energy_budget import MoistureBudgetTracker  # noqa: E402
from legoesm.driver.model_driver import ModelDriver  # noqa: E402


class _Sigma:
    def __init__(self, nlev):
        self.dsigma = jnp.full((nlev,), 1.0 / nlev)

    def pressure_at_half(self, p_s):
        n = self.dsigma.size
        sig = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(self.dsigma)])
        return p_s[..., None] * sig[None, :]


class _Field:
    def __init__(self, data):
        self.data = data


class _State:
    def __init__(self, q_v, p_s):
        self.tracers = {"q_v": _Field(q_v)}
        self.p_s = _Field(p_s)


class _Grid:
    def __init__(self, ncol):
        self.areaCell = jnp.ones((ncol,))


class _Diag:
    def __init__(self):
        self.moisture_tracker = MoistureBudgetTracker()


class _Driver:
    """Only the attributes ``_feed_mpas_moisture_budget`` actually reads."""

    _feed_mpas_moisture_budget = ModelDriver._feed_mpas_moisture_budget

    def __init__(self, ncol=16, nlev=4, voronoi_layout=None):
        self.state = _State(jnp.full((ncol, nlev), 0.01),
                            jnp.full((ncol,), 1.0e5))
        self.sigma = _Sigma(nlev)
        self.grid = _Grid(ncol)
        self._voronoi_layout = voronoi_layout


def _kw(ncol, *, precip_mm_day, hfls_w_m2):
    return {"precip": jnp.full((ncol,), precip_mm_day / 86400.0),
            "hfls": jnp.full((ncol,), hfls_w_m2)}


def test_a_known_imbalance_reaches_the_tracker_with_the_right_size():
    """Evaporation 3 mm/day against rain 2 leaves a 1 mm/day residual."""
    d = _Driver()
    diag = _Diag()
    e_mm_day = 3.0
    hfls = e_mm_day / 86400.0 * constants.L_v      # mm/day -> W/m2
    # Two feeds: the tracker needs a previous sample to form dW/dt, and with a
    # STEADY column that tendency is zero, so the residual is exactly E - P.
    d._feed_mpas_moisture_budget(0.0, diag, _kw(16, precip_mm_day=2.0,
                                                hfls_w_m2=hfls))
    d._feed_mpas_moisture_budget(1.0, diag, _kw(16, precip_mm_day=2.0,
                                                hfls_w_m2=hfls))
    res = diag.moisture_tracker.residual
    assert len(res) == 2, "the tracker was not fed"
    assert res[-1] == pytest.approx(1.0, abs=0.05), (
        f"a 1 mm/day imbalance came back as {res[-1]:.3f}")


def test_a_closed_budget_reads_zero():
    """Rain equal to evaporation on a steady column must leave no residual —
    otherwise the test above would pass on any number the tracker invents."""
    d = _Driver()
    diag = _Diag()
    hfls = 2.0 / 86400.0 * constants.L_v
    for day in (0.0, 1.0):
        d._feed_mpas_moisture_budget(day, diag,
                                     _kw(16, precip_mm_day=2.0,
                                         hfls_w_m2=hfls))
    assert diag.moisture_tracker.residual[-1] == pytest.approx(0.0, abs=0.05)


def test_the_partitioned_lane_publishes_nothing_rather_than_a_local_number():
    """Under a cell partition the tracker's area mean would be rank-local and
    would double-count halo cells. Silence is the correct output."""
    d = _Driver(voronoi_layout=object())
    diag = _Diag()
    d._feed_mpas_moisture_budget(0.0, diag, _kw(16, precip_mm_day=2.0,
                                                hfls_w_m2=100.0))
    assert diag.moisture_tracker.residual == []


def test_a_dry_run_is_skipped_not_crashed():
    d = _Driver()
    d.state.tracers = None
    diag = _Diag()
    d._feed_mpas_moisture_budget(0.0, diag, _kw(16, precip_mm_day=2.0,
                                                hfls_w_m2=100.0))
    assert diag.moisture_tracker.residual == []


def test_withheld_fluxes_are_skipped():
    """A partial diagnostic window publishes no fluxes; the budget must not
    close against a None and record a spurious imbalance."""
    d = _Driver()
    diag = _Diag()
    d._feed_mpas_moisture_budget(0.0, diag, {"precip": None, "hfls": None})
    assert diag.moisture_tracker.residual == []


def test_the_cmor_feed_actually_calls_it():
    """The point of this change is the CALL, not the method.

    Every test above exercises ``_feed_mpas_moisture_budget`` directly and would
    keep passing if the call site vanished — which is exactly how the check came
    to be missing in the first place. This names the enclosing function that
    runs on the MPAS lane and fails if the call is dropped.
    """
    import inspect
    src = inspect.getsource(ModelDriver._feed_mpas_cmip_accumulators)
    assert "_feed_mpas_moisture_budget" in src, (
        "the MPAS CMOR feed no longer records the water budget")
