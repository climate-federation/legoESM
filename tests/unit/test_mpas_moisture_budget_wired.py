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


class _Accum:
    """The flux accumulator, as far as this closure is concerned.

    Its state is load-bearing: an incomplete window means the CMOR slot getter
    would hand back an INSTANTANEOUS diagnostic instead of the window mean, and
    a closure that mixed a mean rainfall with an instantaneous evaporation
    would manufacture an imbalance out of nothing.
    """

    def __init__(self, samples=True, complete=True):
        self._samples, self._complete = samples, complete

    def has_samples(self):
        return self._samples

    def is_complete(self):
        return self._complete


class _Diag:
    def __init__(self):
        self.moisture_tracker = MoistureBudgetTracker()


class _Driver:
    """Only the attributes ``_feed_mpas_moisture_budget`` actually reads."""

    _feed_mpas_moisture_budget = ModelDriver._feed_mpas_moisture_budget

    def __init__(self, ncol=16, nlev=4, voronoi_layout=None, accum=None):
        self.state = _State(jnp.full((ncol, nlev), 0.01),
                            jnp.full((ncol,), 1.0e5))
        self.sigma = _Sigma(nlev)
        self.grid = _Grid(ncol)
        self._voronoi_layout = voronoi_layout
        self._mpas_sfc_accum = _Accum() if accum is None else accum


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
    assert np.isnan(res[0]), (
        "the FIRST sample has no tendency and must report nothing, not zero")
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


def test_an_incomplete_window_is_not_closed_against():
    """A partial window makes the slot getter fall back to instantaneous
    values; mixing those with a mean invents an imbalance."""
    d = _Driver(accum=_Accum(samples=True, complete=False))
    diag = _Diag()
    d._feed_mpas_moisture_budget(0.0, diag, _kw(16, precip_mm_day=2.0,
                                                hfls_w_m2=100.0))
    assert diag.moisture_tracker.residual == []


def test_the_first_sample_after_a_restart_reports_nothing():
    """The tendency baseline lives in memory and no checkpoint carries it, so
    a restarted segment has no closure until its second window. Reporting zero
    there would publish a clean pass at the start of every segment of a chained
    run -- which is most of this campaign's runs."""
    from legoesm.diagnostics.energy_budget import MoistureBudgetTracker as _T
    fresh = _T()          # what a restart hands us
    d = _Driver()
    diag = _Diag()
    diag.moisture_tracker = fresh
    d._feed_mpas_moisture_budget(10.0, diag, _kw(16, precip_mm_day=2.0,
                                                 hfls_w_m2=100.0))
    assert np.isnan(diag.moisture_tracker.residual[-1])


# ---------------------------------------------------------------------------
# Energy-budget closure (#1354).  Same failure mode as the moisture budget:
# the tracker lives in a collector method the MPAS loop never calls, so the
# lane that detonates published NO energy series and the decisive
# leak-vs-imbalance test was never runnable.  These assert the WIRING and the
# arithmetic: a known TOA imbalance on a steady column (dE/dt = 0) must come
# back as exactly that residual, and the partitioned/withheld/absent paths
# must publish nothing rather than a fabricated number.
# ---------------------------------------------------------------------------
from legoesm.diagnostics.energy_budget import EnergyBudgetTracker  # noqa: E402


class _SigmaE:
    """Sigma coord with the full-level pressures the energy tracker needs."""

    def __init__(self, nlev):
        self.dsigma = jnp.full((nlev,), 1.0 / nlev)
        self.sigma_full = (jnp.cumsum(self.dsigma) - 0.5 * self.dsigma)

    def pressure_at_half(self, p_s):
        sig = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(self.dsigma)])
        return p_s[..., None] * sig[None, :]

    def pressure_at_full(self, p_s):
        return p_s[..., None] * self.sigma_full[None, :]


class _StateE:
    def __init__(self, ncol, nlev, moist=True):
        self.T = _Field(jnp.full((ncol, nlev), 250.0))
        self.p_s = _Field(jnp.full((ncol,), 1.0e5))
        self.phis = _Field(jnp.zeros((ncol,)))
        self.u = _Field(jnp.zeros((ncol, nlev)))
        self.tracers = ({"q_v": _Field(jnp.full((ncol, nlev), 0.005))}
                        if moist else None)


class _AccumE:
    """Flux accumulator whose window-mean surface net SW/LW (slots 0/1) the
    energy closure reads.  ``sfc_present=False`` models a run whose surface
    radiation was not diagnosed (mean() returns None) -- the closure must skip.
    """

    def __init__(self, ncol, *, sw_net=0.0, lw_net=0.0,
                 samples=True, complete=True, sfc_present=True):
        self._ncol = ncol
        self._sw_net, self._lw_net = sw_net, lw_net
        self._samples, self._complete, self._sfc = samples, complete, sfc_present

    def has_samples(self):
        return self._samples

    def is_complete(self):
        return self._complete

    def mean(self, slot):
        if not self._sfc:
            return None
        if slot == 0:
            return np.full((self._ncol,), self._sw_net)
        if slot == 1:
            return np.full((self._ncol,), self._lw_net)
        return None


class _DiagE:
    def __init__(self):
        self.energy_tracker = EnergyBudgetTracker()


class _DriverE:
    """Only the attributes ``_feed_mpas_energy_budget`` actually reads."""

    _feed_mpas_energy_budget = ModelDriver._feed_mpas_energy_budget

    def __init__(self, ncol=16, nlev=4, voronoi_layout=None, accum=None,
                 moist=True, sw_net=0.0, lw_net=0.0):
        self.state = _StateE(ncol, nlev, moist=moist)
        self.sigma = _SigmaE(nlev)
        self.grid = _Grid(ncol)
        self._voronoi_layout = voronoi_layout
        self._mpas_sfc_accum = (
            _AccumE(ncol, sw_net=sw_net, lw_net=lw_net)
            if accum is None else accum)


def _kw_e(ncol, nlev, *, rsdt, rsut, rlut, hfss=0.0, hfls=0.0):
    return {"rsdt": jnp.full((ncol,), rsdt),
            "rsut": jnp.full((ncol,), rsut),
            "rlut": jnp.full((ncol,), rlut),
            "hfss": jnp.full((ncol,), hfss),
            "hfls": jnp.full((ncol,), hfls),
            "u_east": jnp.zeros((ncol, nlev)),
            "v_north": jnp.zeros((ncol, nlev))}


def test_energy_a_known_toa_imbalance_reaches_the_tracker():
    """rsdt 340, rsut 100, rlut 200 -> R_TOA = +40; surface flux zero and a
    steady column (dE/dt=0) leaves residual = R_TOA - F_sfc - dE/dt = +40."""
    d = _DriverE()          # sw_net = lw_net = 0 -> F_sfc = 0
    diag = _DiagE()
    kw = _kw_e(16, 4, rsdt=340.0, rsut=100.0, rlut=200.0)
    d._feed_mpas_energy_budget(0.0, diag, kw)
    d._feed_mpas_energy_budget(1.0, diag, kw)
    res = diag.energy_tracker.residual
    toa = diag.energy_tracker.toa_net
    assert len(res) == 2, "the energy tracker was not fed"
    assert toa[-1] == pytest.approx(40.0, abs=1e-6)
    assert res[-1] == pytest.approx(40.0, abs=1e-3), (
        f"a +40 W/m2 imbalance (F_sfc=0) came back as {res[-1]:.3f}")


def test_energy_surface_flux_enters_the_mpas_closure():
    """The MPAS feed must pass the surface net radiation (slots 0/1) AND the
    turbulent fluxes (hfss/hfls) into the closure.  R_TOA=0, sw_net 180,
    lw_net -60, SH 20, LH 60 -> F_sfc = 180-60-20-60 = 40, so a steady column
    reads residual = 0 - 40 - 0 = -40 (would be 0 if the surface term were
    dropped, as it was pre-#1354)."""
    d = _DriverE(sw_net=180.0, lw_net=-60.0)
    diag = _DiagE()
    kw = _kw_e(16, 4, rsdt=340.0, rsut=100.0, rlut=240.0,  # R_TOA = 0
               hfss=20.0, hfls=60.0)
    for day in (0.0, 1.0):
        d._feed_mpas_energy_budget(day, diag, kw)
    assert diag.energy_tracker.sfc_shf[-1] == pytest.approx(20.0, abs=1e-6)
    assert diag.energy_tracker.residual[-1] == pytest.approx(-40.0, abs=1e-3)


def test_energy_a_closed_toa_reads_zero_residual():
    """R_TOA=0 and F_sfc=0 on a steady column must leave no residual, so the
    imbalance test cannot pass on a fabricated number."""
    d = _DriverE()          # F_sfc = 0
    diag = _DiagE()
    kw = _kw_e(16, 4, rsdt=340.0, rsut=100.0, rlut=240.0)   # R_TOA = 0
    for day in (0.0, 1.0):
        d._feed_mpas_energy_budget(day, diag, kw)
    assert diag.energy_tracker.toa_net[-1] == pytest.approx(0.0, abs=1e-6)
    assert diag.energy_tracker.residual[-1] == pytest.approx(0.0, abs=1e-3)


def test_energy_first_sample_residual_is_nan_not_zero():
    """A fresh tracker (every restart segment) has no tendency yet, so the
    first residual is NaN.  Publishing 0 would read as a clean pass at the
    start of every chained-run segment."""
    d = _DriverE()
    diag = _DiagE()
    d._feed_mpas_energy_budget(0.0, diag,
                               _kw_e(16, 4, rsdt=340.0, rsut=100.0, rlut=200.0))
    assert np.isnan(diag.energy_tracker.residual[-1])


def test_energy_column_energy_is_finite_and_earthlike():
    """The column MSE the tendency is differenced from must be a sane positive
    energy (~ a few GJ/m2), not a NaN or a fabricated zero."""
    d = _DriverE()
    diag = _DiagE()
    d._feed_mpas_energy_budget(0.0, diag,
                               _kw_e(16, 4, rsdt=340.0, rsut=100.0, rlut=240.0))
    E = diag.energy_tracker.column_energy[-1]
    assert np.isfinite(E) and 1.0e9 < E < 5.0e9, (
        f"column energy {E:.3e} J/m2 is outside the Earth-like range")


def test_energy_a_dry_run_still_closes():
    """Unlike moisture, the energy budget is defined on a dry column; it must
    record, not skip."""
    d = _DriverE(moist=False)
    diag = _DiagE()
    for day in (0.0, 1.0):
        d._feed_mpas_energy_budget(day, diag,
                                   _kw_e(16, 4, rsdt=340.0, rsut=100.0,
                                         rlut=200.0))
    assert len(diag.energy_tracker.residual) == 2
    assert diag.energy_tracker.residual[-1] == pytest.approx(40.0, abs=1e-3)


def test_energy_absent_surface_radiation_is_skipped():
    """If the surface net radiation was not diagnosed this window (accumulator
    returns None for slots 0/1) there is no closure to publish -- skip rather
    than invent one from a partial surface flux."""
    d = _DriverE(accum=_AccumE(16, sfc_present=False))
    diag = _DiagE()
    d._feed_mpas_energy_budget(0.0, diag,
                               _kw_e(16, 4, rsdt=340.0, rsut=100.0, rlut=200.0))
    assert diag.energy_tracker.residual == []


def test_energy_the_partitioned_lane_publishes_nothing():
    """Under a cell partition the area mean is rank-local; silence is right."""
    d = _DriverE(voronoi_layout=object())
    diag = _DiagE()
    d._feed_mpas_energy_budget(0.0, diag,
                               _kw_e(16, 4, rsdt=340.0, rsut=100.0, rlut=200.0))
    assert diag.energy_tracker.residual == []


def test_energy_withheld_toa_fluxes_are_skipped():
    """A run without radiation hands None TOA fluxes; the budget must not close
    against them."""
    d = _DriverE()
    diag = _DiagE()
    d._feed_mpas_energy_budget(0.0, diag,
                               {"rsdt": None, "rsut": None, "rlut": None})
    assert diag.energy_tracker.residual == []


def test_energy_an_incomplete_window_is_not_closed_against():
    d = _DriverE(accum=_AccumE(16, samples=True, complete=False))
    diag = _DiagE()
    d._feed_mpas_energy_budget(0.0, diag,
                               _kw_e(16, 4, rsdt=340.0, rsut=100.0, rlut=200.0))
    assert diag.energy_tracker.residual == []


def test_energy_the_cmor_feed_actually_calls_it():
    """The point is the CALL: every test above exercises the method directly
    and would keep passing if the call site vanished -- the exact way the check
    came to be missing.  Name the enclosing function that runs on the lane and
    fail if the call is dropped."""
    import inspect
    src = inspect.getsource(ModelDriver._feed_mpas_cmip_accumulators)
    assert "_feed_mpas_energy_budget" in src, (
        "the MPAS CMOR feed no longer records the energy budget")


def test_energy_backfill_uses_the_current_interval_not_a_stale_one():
    """P1a (codex): the lean loop appends a NaN placeholder, feeds the tracker,
    then backfills.  The backfill must fill the row with THIS interval's fresh
    value when the feed added a sample, and leave the NaN when it did not (a
    withheld window) -- never a stale previous value."""
    bf = ModelDriver._backfill_energy_row

    class _Trk:
        def __init__(self):
            self.residual, self.toa_net, self.column_energy = [], [], []

        def add(self, r, t, c):
            self.residual.append(r); self.toa_net.append(t)
            self.column_energy.append(c)

    trk = _Trk()
    ts = {"energy_residual": [], "energy_toa_net": [], "energy_column": []}

    # Interval 1: placeholder appended, feed adds a sample -> backfilled fresh.
    for k in ts:
        ts[k].append(float("nan"))
    n0 = len(trk.residual)
    trk.add(-40.0, 0.0, 2.6e9)
    bf(ts, trk, n0)
    assert ts["energy_residual"][-1] == pytest.approx(-40.0)
    assert ts["energy_column"][-1] == pytest.approx(2.6e9)

    # Interval 2: placeholder appended, feed WITHHELD (no new sample) -> NaN
    # stands, and does NOT copy interval 1's -40.
    for k in ts:
        ts[k].append(float("nan"))
    n0 = len(trk.residual)
    bf(ts, trk, n0)          # tracker unchanged
    assert np.isnan(ts["energy_residual"][-1]), "a withheld window must stay NaN"

    # Interval 3 (the FINAL interval): a sample is added and captured -- the
    # pre-fix bug dropped the last interval because sampling ran before the feed.
    for k in ts:
        ts[k].append(float("nan"))
    n0 = len(trk.residual)
    trk.add(-38.0, 1.0, 2.61e9)
    bf(ts, trk, n0)
    assert ts["energy_residual"][-1] == pytest.approx(-38.0)
    assert len(ts["energy_residual"]) == 3
