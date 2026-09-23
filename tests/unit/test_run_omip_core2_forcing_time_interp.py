"""Gap 12: monthly forcing climatologies can interpolate in time like ORCA1.

From the oracle's OWN namelist_cfg (not namelist_ref), column 4 of each sn_*
row is ln_tint: sn_chl:170, sn_sss:185 and sn_rnf:199 are all .true. with
freq -1 (monthly), while the 6-hourly winds and the radiation fields at
lines 148-152 are .FALSE. So the faithful change is to interpolate exactly
these three and to keep HOLDING everything else.
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

RUNNER = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "run" / "run_omip_core2.py")


def _mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_omip_runner_fti", RUNNER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DAY = 86400.0


def _field():
    """12 months whose values are the month index, so a sample reads back as
    a fractional month and the interpolation is legible."""
    return np.arange(12, dtype=float).reshape(12, 1)


# --- the default must change nothing ----------------------------------------

def test_hold_is_identical_to_the_previous_indexing():
    m, f = _mod(), _field()
    for day in range(0, 365, 7):
        step = int(day * DAY / 300.0)
        assert m.month_sample(f, step, 300.0, "hold") == \
            f[m._runoff_month_idx(step, 300.0)]


def test_none_passes_through():
    assert _mod().month_sample(None, 0, 300.0, "linear") is None


def test_unknown_mode_raises():
    with pytest.raises(SystemExit, match="unknown forcing"):
        _mod().month_sample(_field(), 0, 300.0, "nearest")


# --- the interpolation itself -----------------------------------------------

def test_weight_is_zero_exactly_at_a_month_centre():
    """The defining property: at a record's own centre the interpolation must
    return that record untouched, or every month is biased toward its
    neighbour."""
    m = _mod()
    for mon in range(12):
        day = float(m._MONTH_MID[mon])
        step = int(round(day * DAY / 300.0))
        i0, i1, w = m._month_interp_weights(step, 300.0)
        near = w if i0 == mon else (1.0 - w)
        assert near == pytest.approx(0.0, abs=2e-3), (mon, i0, i1, w)


def test_record_centres_are_the_actual_middles_of_the_months():
    """Pins WHERE the centres are, not merely that the weight vanishes there.

    Found by a planted defect: replacing the centres with month ENDS left
    test_weight_is_zero_exactly_at_a_month_centre green, because that test
    only asks for w=0 at whatever the code calls a centre. A half-month phase
    error in every monthly forcing field would have passed.
    """
    mid = _mod()._MONTH_MID
    assert mid[0] == pytest.approx(15.5)    # January, days 0-31
    assert mid[1] == pytest.approx(45.0)    # February, 31 + 28/2
    # July: cumulative day 212 at its end, minus half of 31.
    assert mid[6] == pytest.approx(196.5)
    assert mid[11] == pytest.approx(349.5)  # December, 365 - 31/2
    assert np.all(np.diff(mid) > 0)


def test_it_wraps_december_to_january():
    """A climatology is periodic: 1 January sits BETWEEN the December and
    January centres, so the pair must be (11, 0), not (0, 1)."""
    m = _mod()
    i0, i1, w = m._month_interp_weights(0, 300.0)   # day 0.0
    assert (i0, i1) == (11, 0), (i0, i1)
    assert 0.0 < w < 1.0


def test_interpolated_value_is_strictly_between_its_two_records():
    m, f = _mod(), _field()
    # A day inside a month but away from its centre must not equal any record.
    day = float(m._MONTH_MID[5] + 8.0)
    step = int(day * DAY / 300.0)
    got = float(m.month_sample(f, step, 300.0, "linear")[0])
    lo, hi = 5.0, 6.0
    assert lo < got < hi, got
    assert got != pytest.approx(float(m.month_sample(f, step, 300.0, "hold")[0]))


def test_linear_mode_is_continuous_across_every_month_boundary():
    """The whole point of ln_tint: no step discontinuity. Hold mode HAS one,
    which is what makes this test non-vacuous.

    The fixture must be PERIODIC. A field equal to the month index is not: it
    sweeps 11 -> 0 across the year wrap, and interpolating that ramp produces
    a legitimate 11/31 = 0.355 per day slope that looks like a discontinuity
    and is not one. My first version of this test used the index field and
    failed for exactly that reason -- the fixture was wrong, not the sampler.
    """
    m = _mod()
    month = np.arange(12, dtype=float)
    f = np.sin(2.0 * np.pi * month / 12.0).reshape(12, 1)
    dt = 300.0
    worst_lin = worst_hold = 0.0
    prev_l = prev_h = None
    for k in range(0, int(365 * DAY / dt), int(DAY / dt)):
        l = float(m.month_sample(f, k, dt, "linear")[0])
        h = float(m.month_sample(f, k, dt, "hold")[0])
        if prev_l is not None:
            worst_lin = max(worst_lin, abs(l - prev_l))
            worst_hold = max(worst_hold, abs(h - prev_h))
        prev_l, prev_h = l, h
    # Largest month-to-month change of sin is 0.5 over ~29 days -> ~0.018/day.
    assert worst_lin < 0.03, f"linear jumped {worst_lin}"
    # Hold steps the WHOLE month-to-month difference in a single day.
    assert worst_hold > 0.4, f"hold should step, got {worst_hold}"


def test_interpolation_reweights_months_by_their_neighbours_lengths():
    """What linear ACTUALLY does to an annual integral, stated exactly.

    My first version asserted the annual mean of a month-index field stays
    5.5, which codex showed passes under 'hold' too -- it could not establish
    anything about conservation. Both reviewers then converged on the real
    statement: linear interpolation between mid-month anchors weights month k
    by (M[k-1] + 2*M[k] + M[k+1]) / 4 days instead of M[k]. For February that
    is (31 + 56 + 31)/4 = 29.5 equivalent days against 28 held -- the number
    codex quoted and the formula GLM gave, independently.

    So the annual integral is NOT exactly preserved for unequal month lengths.
    It is close (the reweighting is a few percent on the short months and
    cancels to well under a percent of the year), but a runoff climatology
    adopted with 'linear' does deliver a slightly different annual freshwater
    total, and that belongs in the record rather than being assumed away.
    """
    m = _mod()
    days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], float)
    eff = (np.roll(days, 1) + 2 * days + np.roll(days, -1)) / 4.0
    assert eff[1] == pytest.approx(29.5)           # February
    assert eff.sum() == pytest.approx(365.0)       # the YEAR still closes

    # And the sampler agrees with that reweighting: integrate an indicator for
    # February and compare against its effective length.
    ind = np.zeros((12, 1)); ind[1] = 1.0
    dt = 3600.0
    n = int(365 * DAY / dt)
    lin_days = sum(float(m.month_sample(ind, k, dt, "linear")[0])
                   for k in range(n)) * dt / DAY
    hold_days = sum(float(m.month_sample(ind, k, dt, "hold")[0])
                    for k in range(n)) * dt / DAY
    assert hold_days == pytest.approx(28.0, abs=0.05), hold_days
    assert lin_days == pytest.approx(29.5, abs=0.1), lin_days


# --- the wiring -------------------------------------------------------------

def test_all_six_monthly_call_sites_use_the_sampler():
    """Three fields x two lanes. Counted over the AST so a commented-out call
    cannot satisfy it -- the substring form of this check was shown to accept
    inert wiring on gap 11."""
    calls = [n for n in ast.walk(ast.parse(RUNNER.read_text()))
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "month_sample"]
    assert len(calls) == 6, f"expected 6 wired call sites, found {len(calls)}"


def test_flag_round_trips_and_defaults_to_hold():
    p = _mod()._build_arg_parser()
    assert p.parse_args([]).forcing_time_interp == "hold"
    assert p.parse_args(["--forcing-time-interp",
                         "linear"]).forcing_time_interp == "linear"


@pytest.mark.parametrize("dest", ("forcing_time_interp", "ice_exchange"))
def test_the_flag_is_reachable_on_the_fesom_lane(dest):
    """A flag can be fully wired and still REFUSED.

    The FESOM lane raises SystemExit for any dest whose value differs from its
    default and is not in _FESOM_WIRED_DESTS. Gap 12's sampler and gap 11's
    exchange set are both wired into that lane's own code, but neither was
    listed, so both were unreachable there (codex, MAJOR). Guard both.
    """
    assert dest in _mod()._FESOM_WIRED_DESTS


def test_the_oracle_rows_are_cited():
    """These line numbers are the evidence for the whole change; if someone
    edits the rationale away, the claim loses its source."""
    src = RUNNER.read_text()
    assert "ln_tint" in src and "namelist_cfg:170" in src
